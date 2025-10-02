from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC, CLIPProcessor, CLIPModel, Wav2Vec2Model
import torch
import torch.nn as nn
from PIL import Image
import torchaudio

class AudioEncoder(nn.Module):
    """
    Wav2Vec2を使用した音声エンコーダー
    
    入力:
        data: dict with keys:
            - "wav": List[np.ndarray] - 各要素は(T,)の1次元音声波形
    
    出力:
        audio_features: Tensor[B, seq_len, 768] - 音声特徴量
    
    Note:
        - 入力音声は16kHzを想定
        - 異なる長さの音声は自動的にパディングされる
        - 事前学習済みWav2Vec2モデルを使用（CTCヘッドなし）
    """
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Wav2Vec2Modelを使用（CTCヘッドなし、メモリ効率化）
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
        # 語彙サイズを取得（CTCモデルから一時的に取得）
        temp_ctc_model = AutoModelForCTC.from_pretrained(model_name)
        self.vocab_size = temp_ctc_model.config.vocab_size
        del temp_ctc_model
        
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
            # 音声の前処理（自動パディング）
            input_values = self.processor(
                wav,
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).input_values.to(device)

            
        except Exception as e:
            print(f"Error in audio processing: {e}")
            print(f"  Input types: {[type(w) for w in wav]}")
            print(f"  Input shapes: {[w.shape if hasattr(w, 'shape') else len(w) for w in wav]}")
            raise
        
        # 特徴抽出
        audio_outputs = self.model(
            input_values,
            output_hidden_states=False,
            return_dict=True
        )
        audio_features = audio_outputs.last_hidden_state  # [B, seq_len, 768]
        
        # NaN/Infチェック
        if torch.isnan(audio_features).any():
            # print(f"wav: {wav}, wav shape: {[w.shape for w in wav]}")
            # print(f"Input values: {input_values}, Input values shape: {input_values.shape}")
            # print(f"Audio features: {audio_features}, Audio features shape: {audio_features.shape}")
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
        keys = torch.cat([audio_k, vision_k], dim=1)     # [B, seq_len+1, hidden_dim]
        values = torch.cat([audio_v, vision_v], dim=1)   # [B, seq_len+1, hidden_dim]
        
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
        "image": [image],       # List形式
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


if __name__ == "__main__":
    demo()