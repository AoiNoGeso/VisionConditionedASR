from model import VisionConditionedASR
import torchaudio
import torch
from PIL import Image
import numpy as np
import os

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