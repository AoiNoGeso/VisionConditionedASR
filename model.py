from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC, CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
from PIL import Image
import torchaudio
import numpy as np
import os

class AudioEncoder(nn.Module):
    """
    Wav2Vec2を使用した音声エンコーダー
    
    入力:
        data: dict with keys:
            - "wav": List[np.ndarray] - 各要素は(T,)の1次元音声波形
    
    出力:
        audio_features: Tensor[B, seq_len, 768] - 音声特徴量
    """
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        self.processor = AutoProcessor.from_pretrained(model_name, force_download=True)
        
        # AutoModelForCTCを使用
        self.model = AutoModelForCTC.from_pretrained(model_name, force_download=True)

        # 💡修正: masked_spec_embedを無条件で再初期化
        if hasattr(self.model, 'wav2vec2') and hasattr(self.model.wav2vec2, 'masked_spec_embed'):
            if self.model.wav2vec2.masked_spec_embed is not None:
                print("[AudioEncoder] Re-initializing masked_spec_embed to avoid NaN...")
                
                # 小さい範囲の一様分布で初期化
                nn.init.uniform_(
                    self.model.wav2vec2.masked_spec_embed.data,
                    a=-0.01,
                    b=0.01
                )
                
                # 再初期化後の確認
                if torch.isnan(self.model.wav2vec2.masked_spec_embed).any():
                    raise RuntimeError("🚨 CRITICAL: Failed to initialize masked_spec_embed!")
                
                print(f"[AudioEncoder] ✓ masked_spec_embed initialized successfully")
                print(f"  Range: [{self.model.wav2vec2.masked_spec_embed.min().item():.6f}, "
                      f"{self.model.wav2vec2.masked_spec_embed.max().item():.6f}]")
        
        self.vocab_size = self.model.config.vocab_size
        print(f"[AudioEncoder] Vocab size: {self.vocab_size}")
        
        # デバイスの保存
        self._device = device
        
    def forward(self, data):
        """
        Args:
            data: dict with key "wav" (List[np.ndarray])
        
        Returns:
            audio_features: Tensor[B, seq_len, 768]
        """
        wav = data["wav"]
        
        # 入力チェック
        if not wav or len(wav) == 0:
            raise ValueError("Empty audio input received")
        
        for i, w in enumerate(wav):
            if len(w) == 0:
                raise ValueError(f"Audio sample {i} has zero length")
        
        # デバイスを取得
        if self._device is not None:
            device = self._device
        else:
            device = next(self.model.parameters()).device

        
        try:
            # 💡修正: 音声の前処理（attention_maskを取得）
            processed = self.processor(
                wav,
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            input_values = processed.input_values.to(device)
            attention_mask = processed.attention_mask.to(device) if hasattr(processed, 'attention_mask') else None
            
            # 💡追加: 入力値の健全性チェック
            if torch.isnan(input_values).any() or torch.isinf(input_values).any():
                raise RuntimeError("Input values contain NaN or Inf after processing")
            
            # 💡追加: 入力値の範囲チェック（異常な値をクリッピング）
            input_values = torch.clamp(input_values, min=-10.0, max=10.0)
            
        except Exception as e:
            print(f"Error in audio processing: {e}")
            print(f"  Input types: {[type(w) for w in wav]}")
            print(f"  Input shapes: {[w.shape if hasattr(w, 'shape') else len(w) for w in wav]}")
            raise
        
        # 💡修正: attention_maskを渡して特徴抽出
        audio_outputs = self.model.wav2vec2(
            input_values,
            attention_mask=attention_mask,  # パディング位置を明示
            output_hidden_states=False,
            return_dict=True
        )
        audio_features = audio_outputs.last_hidden_state  # [B, seq_len, 768]
        
        # NaN/Infチェック
        if torch.isnan(audio_features).any():
            print(f"\n🚨 CRITICAL: NaN detected in audio features!")
            print(f"  Input values range: [{input_values.min():.4f}, {input_values.max():.4f}]")
            print(f"  Input values shape: {input_values.shape}")
            print(f"  Attention mask: {attention_mask}")
            raise RuntimeError("Audio features contain NaN values")
        if torch.isinf(audio_features).any():
            raise RuntimeError("Audio features contain Inf values")
        
        return audio_features


class VisionEncoder(nn.Module):
    """
    CLIPを使用した画像エンコーダー
    
    入力:
        data: dict with keys:
            - "image": List[PIL.Image.Image] - RGB画像のリスト
    
    出力:
        image_features: Tensor[B, 512] - 画像特徴量（グローバル表現）
    
    Note:
        - 画像は自動的にCLIPの入力サイズにリサイズされる
        - 事前学習済みCLIP ViT-B/16モデルを使用
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch16", device=None):
        super().__init__()
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        
        # デバイスの保存
        self._device = device
        
    def forward(self, data):
        """
        Args:
            data: dict with key "image" (List[PIL.Image])
        
        Returns:
            image_features: Tensor[B, 512]
        """
        images = data["image"]
        
        # 入力チェック
        if not images or len(images) == 0:
            raise ValueError("Empty image input received")
        
        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                raise TypeError(f"Image {i} is not a PIL.Image object: {type(img)}")
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"Image {i} has invalid size: {img.size}")
        
        # デバイスを取得
        if self._device is not None:
            device = self._device
        else:
            device = next(self.model.parameters()).device
        
        try:
            # 画像の前処理
            inputs = self.processor(images=images, return_tensors="pt")
            
            # デバイスに移動
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
        except Exception as e:
            print(f"Error in image processing: {e}")
            print(f"  Input types: {[type(img) for img in images]}")
            print(f"  Input sizes: {[img.size for img in images]}")
            raise
        
        # 特徴抽出
        image_features = self.model.get_image_features(**inputs)  # [B, 512]
        
        # NaN/Infチェック
        if torch.isnan(image_features).any():
            raise RuntimeError("Image features contain NaN values")
        if torch.isinf(image_features).any():
            raise RuntimeError("Image features contain Inf values")
        
        return image_features


class CrossAttention(nn.Module):
    """
    音声と画像特徴を統合するクロスアテンション層
    
    Args:
        audio_dim: 音声特徴の次元数
        vision_dim: 画像特徴の次元数
        hidden_dim: アテンション層の隠れ次元数
        num_heads: マルチヘッドアテンションのヘッド数
    """
    
    def __init__(self, audio_dim=768, vision_dim=512, hidden_dim=256, num_heads=2):
        super().__init__()
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 音声特徴の射影
        self.audio_query = nn.Linear(audio_dim, hidden_dim)
        self.audio_key = nn.Linear(audio_dim, hidden_dim)
        self.audio_value = nn.Linear(audio_dim, hidden_dim)
        
        # 画像特徴の射影
        self.vision_key = nn.Linear(vision_dim, hidden_dim)
        self.vision_value = nn.Linear(vision_dim, hidden_dim)
        
        # マルチヘッドアテンション
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # 出力層
        self.output_proj = nn.Linear(hidden_dim, audio_dim)
        self.layer_norm = nn.LayerNorm(audio_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, audio_features, vision_features):
        """
        Args:
            audio_features: [batch, seq_len, audio_dim] 
            vision_features: [batch, vision_dim]
        
        Returns:
            enhanced_audio_features: [batch, seq_len, audio_dim]
        """
        batch_size, seq_len, audio_dim = audio_features.size()
        vision_features = vision_features.unsqueeze(1)  # [B, 1, vision_dim]
        
        # 特徴の射影
        audio_q = self.audio_query(audio_features)  # [B, seq_len, hidden_dim]
        audio_k = self.audio_key(audio_features)    # [B, seq_len, hidden_dim]
        audio_v = self.audio_value(audio_features)  # [B, seq_len, hidden_dim]
        
        vision_k = self.vision_key(vision_features)   # [B, 1, hidden_dim]
        vision_v = self.vision_value(vision_features) # [B, 1, hidden_dim]
        
        # クロスアテンションのためのkey/valueの結合
        keys = torch.cat([audio_k, vision_k], dim=1)      # [B, seq_len+1, hidden_dim]
        values = torch.cat([audio_v, vision_v], dim=1)    # [B, seq_len+1, hidden_dim]
        
        # マルチヘッドアテンション（attention weightsは保存しない）
        attn_output, _ = self.multihead_attn(
            query=audio_q,
            key=keys,
            value=values,
            need_weights=False  # メモリ削減
        )
        
        # 出力射影と残差接続
        projected_output = self.output_proj(attn_output)
        output = self.layer_norm(audio_features + self.dropout(projected_output))
        
        return output


class VisionConditionedASR(nn.Module):
    """
    視覚情報で条件付けされた音声認識モデル
    
    入力:
        data: dict with keys:
            - "wav": List[np.ndarray] - 音声波形のリスト
            - "image": List[PIL.Image.Image] - 画像のリスト
    
    出力:
        logits: Tensor[B, seq_len, vocab_size] - CTC用のロジット
    
    アーキテクチャ:
        1. AudioEncoder: 音声 -> [B, seq_len, 768]
        2. VisionEncoder: 画像 -> [B, 512]
        3. CrossAttention: 音声と画像の統合
        4. Classifier: [B, seq_len, vocab_size]へ射影
    
    Args:
        vocab_size: 出力語彙サイズ（Noneで自動取得）
        hidden_dim: Cross-Attentionの隠れ層次元
        num_heads: マルチヘッドアテンションのヘッド数
        device: 実行デバイス（Noneでパラメータから自動取得）
    """
    
    def __init__(self, vocab_size=None, hidden_dim=256, num_heads=2, device=None):
        super().__init__()
        self._device = device
        
        # エンコーダーの初期化
        self.audio_encoder = AudioEncoder(device=device)
        self.vision_encoder = VisionEncoder(device=device)
        
        # 語彙サイズの自動取得
        if vocab_size is None:
            vocab_size = self.audio_encoder.vocab_size
            print(f"[Model] Auto-detected vocab_size: {vocab_size}")
        
        self.vocab_size = vocab_size
        
        # クロスアテンション層
        self.cross_attention = CrossAttention(
            audio_dim=768,
            vision_dim=512,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # 分類器
        self.classifier = nn.Linear(768, vocab_size)
        
    def forward(self, data):
        """
        Args:
            data: dict with keys "wav" and "image"
        
        Returns:
            logits: Tensor[B, seq_len, vocab_size]
        """
        # 音声特徴抽出: [B, seq_len, 768]
        audio_features = self.audio_encoder(data)
        
        # 画像特徴抽出: [B, 512]
        vision_features = self.vision_encoder(data)
        
        # クロスアテンション: [B, seq_len, 768]
        enhanced_audio = self.cross_attention(audio_features, vision_features)
        
        # 中間特徴を削除（メモリ削減）
        del audio_features, vision_features
        
        # 分類: [B, seq_len, vocab_size]
        output_logits = self.classifier(enhanced_audio)
        
        return output_logits


def demo():
    """デモ実行関数"""
    print("="*60)
    print("Vision-Conditioned ASR Demo")
    print("="*60)
    
    # モデルの初期化
    print("\n[1/4] Initializing model...")
    avsr = VisionConditionedASR()
    avsr.eval()  # 評価モードに設定
    
    # データパス
    wav_path = "../../Datasets/SpokenCOCO/wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav"
    img_path = "../../Datasets/stair_captions/images/val2014/COCO_val2014_000000325114.jpg"
    
    # 音声データ読み込み
    print("\n[2/4] Loading audio data...")
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    print(f"  Audio shape: {wav.shape}, Sample rate: 16000Hz")
    
    # 画像データ読み込み
    print("\n[3/4] Loading image data...")
    image = Image.open(img_path).convert('RGB')
    print(f"  Image size: {image.size}, Mode: {image.mode}")
    
    # データセット作成（バッチ形式）
    sample = {
        "wav": [wav.numpy()],  # List形式
        "image": [image],        # List形式
        "text": "A URINAL IN A PUBLIC RESTROOM NEAR A WOODEN TABLE"
    }
    
    # 推論実行
    print("\n[4/4] Running inference...")
    with torch.no_grad():
        # 各コンポーネントの出力確認
        audio_outputs = avsr.audio_encoder(data=sample)
        vision_outputs = avsr.vision_encoder(data=sample)
        asr_outputs = avsr(data=sample)
    
    print(f"\n{'='*60}")
    print("Output Shapes:")
    print(f"{'='*60}")
    print(f"  Audio features:  {audio_outputs.shape}")
    print(f"  Vision features: {vision_outputs.shape}")
    print(f"  ASR logits:      {asr_outputs.shape}")
    
    # デコーディング処理
    print(f"\n{'='*60}")
    print("Decoding Results:")
    print(f"{'='*60}")
    
    tokenizer = avsr.audio_encoder.tokenizer
    
    # logitsから予測されたトークンIDを取得
    predicted_ids = torch.argmax(asr_outputs, dim=-1)  # [B, seq_len]
    
    # CTC blank tokenのID（通常は0）
    blank_token_id = 0
    
    for i, pred_ids in enumerate(predicted_ids):
        pred_ids = pred_ids.cpu().numpy()
        
        # CTCデコーディング
        pred_tokens = []
        prev_token = None
        
        for token_id in pred_ids:
            # blank tokenをスキップ
            if token_id == blank_token_id:
                prev_token = None
                continue
            
            # 連続する同じトークンは1つだけ保持
            if token_id != prev_token:
                pred_tokens.append(token_id)
            
            prev_token = token_id
        
        # デコード
        transcription = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        
        print(f"\nSample {i}:")
        print(f"  Ground Truth: {sample['text']}")
        print(f"  Prediction:   {transcription}")
        print(f"  Note: Model is untrained, output is random")
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print(f"{'='*60}\n")


def demo_batch():
    """バッチ処理によるデモ実行関数（複数の音声と画像を同時に処理）"""
    print("="*60)
    print("Vision-Conditioned ASR Batch Demo")
    print("="*60)
    
    # モデルの初期化
    print("\n[1/5] Initializing model...")
    avsr = VisionConditionedASR()
    avsr.eval()  # 評価モードに設定
    
    sample_files = [
        {
            "wav": "../../Datasets/SpokenCOCO/wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav",
            "img": "../../Datasets/stair_captions/images/val2014/COCO_val2014_000000325114.jpg",
            "text": "A URINAL IN A PUBLIC RESTROOM NEAR A WOODEN TABLE"
        },
        # 💡改善: 実際には異なるファイルを使用することを推奨
        # 以下は2つ目のサンプル例（実際のパスに置き換える）
        {
            "wav": "../../Datasets/SpokenCOCO/wavs/val/0/m1a5mox83rrx60-3V5Q80FXIXRDGZWLAJ5EEBXFON723D_297698_737627.wav",
            "img": "../../Datasets/stair_captions/images/val2014/COCO_val2014_000000297698.jpg",
            "text": "THE SKIER TAKES OFF DOWN THE STEEP HILL"
        }
    ]
    
    # 💡追加: ファイル存在チェックと音声/画像のロード
    wavs = []
    images = []
    texts = []
    
    print("\n[2/5] Loading samples...")
    for i, sample_file in enumerate(sample_files):
        try:
            # 💡追加: ファイル存在チェック
            if not os.path.exists(sample_file["wav"]):
                print(f"  ⚠️  Sample {i+1}: Audio file not found, skipping...")
                continue
            if not os.path.exists(sample_file["img"]):
                print(f"  ⚠️  Sample {i+1}: Image file not found, skipping...")
                continue
            
            # 音声読み込み
            wav, sr = torchaudio.load(sample_file["wav"])
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0).numpy()
            
            # 画像読み込み
            image = Image.open(sample_file["img"]).convert('RGB')
            
            wavs.append(wav)
            images.append(image)
            texts.append(sample_file["text"])
            
            # 💡追加: 各サンプルの情報を表示
            print(f"  ✓ Sample {i+1}: Audio shape={wav.shape}, Image size={image.size}")
            
        except Exception as e:
            print(f"  ✗ Sample {i+1}: Error loading - {e}")
            continue
    
    # 💡追加: ロードされたサンプルが0の場合はエラー
    if len(wavs) == 0:
        print("\n❌ No valid samples loaded. Please check file paths.")
        return
    
    # 💡追加: 音声長の統計情報を表示
    wav_lengths = [len(w) for w in wavs]
    print(f"\n[3/5] Audio length statistics:")
    print(f"  Min length: {min(wav_lengths):,} samples ({min(wav_lengths)/16000:.2f}s)")
    print(f"  Max length: {max(wav_lengths):,} samples ({max(wav_lengths)/16000:.2f}s)")
    print(f"  Mean length: {np.mean(wav_lengths):,.0f} samples ({np.mean(wav_lengths)/16000:.2f}s)")
    print(f"  → Padding will be applied to max length")
    
    # バッチデータセット作成
    batch_sample = {
        "wav": wavs,
        "image": images,
        "text": texts
    }
    
    # 推論実行
    print(f"\n[4/5] Running batch inference (Batch Size: {len(batch_sample['wav'])})...")
    try:
        with torch.no_grad():
            # 各コンポーネントの出力確認
            audio_outputs = avsr.audio_encoder(data=batch_sample)
            vision_outputs = avsr.vision_encoder(data=batch_sample)
            asr_outputs = avsr(data=batch_sample)
        
        # --- 出力形状の確認 ---
        print(f"\n{'='*60}")
        print("Batch Output Shapes:")
        print(f"{'='*60}")
        print(f"  Audio features:  {audio_outputs.shape}  # [B, seq_len, 768]")
        print(f"  Vision features: {vision_outputs.shape}  # [B, 512]")
        print(f"  ASR logits:      {asr_outputs.shape}    # [B, seq_len, vocab_size]")
        print(f"  Batch size (B):  {asr_outputs.shape[0]}")
        
        # 💡追加: NaN/Infチェック
        if torch.isnan(asr_outputs).any():
            print("\n⚠️  WARNING: NaN detected in ASR outputs!")
        if torch.isinf(asr_outputs).any():
            print("\n⚠️  WARNING: Inf detected in ASR outputs!")
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # --- デコーディング処理 ---
    print(f"\n[5/5] Decoding Batch Results...")
    print(f"{'='*60}")
    
    tokenizer = avsr.audio_encoder.tokenizer
    
    # logitsから予測されたトークンIDを取得
    predicted_ids = torch.argmax(asr_outputs, dim=-1)  # [B, seq_len]
    
    # CTC blank tokenのID（通常は0）
    blank_token_id = 0
    
    for i, pred_ids in enumerate(predicted_ids):
        pred_ids = pred_ids.cpu().numpy()
        
        # CTCデコーディング
        pred_tokens = []
        prev_token = None
        
        for token_id in pred_ids:
            # blank tokenをスキップ
            if token_id == blank_token_id:
                prev_token = None
                continue
            
            # 連続する同じトークンは1つだけ保持
            if token_id != prev_token:
                pred_tokens.append(token_id)
            
            prev_token = token_id
        
        # デコード
        transcription = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        
        print(f"\nSample {i+1} / {len(predicted_ids)}:")
        print(f"  Audio length: {wav_lengths[i]:,} samples ({wav_lengths[i]/16000:.2f}s)")
        print(f"  Ground Truth: {batch_sample['text'][i]}")
        print(f"  Prediction:   {transcription}")
        print(f"  Note: Model is untrained, output is random")
    
    print(f"\n{'='*60}")
    print("Batch Demo completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # demo()         # 単一データ版
    demo_batch()    # バッチ処理版

