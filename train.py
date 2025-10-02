import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from model import VisionConditionedASR
from dataloader import create_dataloader


@dataclass
class TrainingConfig:
    """学習設定"""
    # データセット設定
    train_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_train.json"
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    # モデル設定
    vocab_size: Optional[int] = None  # Noneで自動取得
    hidden_dim: int = 256
    num_heads: int = 2
    
    # 学習設定
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # データローダー設定
    num_workers: int = 4
    max_audio_length: float = 10.0  # 秒
    validate_files: bool = True
    
    # 層凍結設定
    freeze_audio_encoder: bool = True
    freeze_vision_encoder: bool = True
    freeze_cross_attention: bool = False
    
    # 学習スケジュール
    warmup_steps: int = 1000
    
    # 保存設定
    checkpoint_dir: str = "../checkpoints"
    save_epoch: int = 1  # エポックごと
    
    # デバイス設定
    device: str = "cuda:1"  # "cuda:0", "cuda:1", "cpu"
    
    # ログ設定
    log_step: int = 100  # ステップごと
    validate_epoch: int = 1  # エポックごと


def freeze_layers(model: VisionConditionedASR, config: TrainingConfig):
    """
    指定された層を凍結
    
    Args:
        model: VisionConditionedASRモデル
        config: 学習設定
    """
    print("\n" + "="*60)
    print("Layer Freeze Configuration")
    print("="*60)
    
    # Audio Encoderの凍結
    if config.freeze_audio_encoder:
        for param in model.audio_encoder.model.parameters():
            param.requires_grad = False
        print("✓ Audio Encoder (Wav2Vec2):    FROZEN")
    else:
        print("✓ Audio Encoder (Wav2Vec2):    Trainable")
    
    # Vision Encoderの凍結
    if config.freeze_vision_encoder:
        for param in model.vision_encoder.model.vision_model.parameters():
            param.requires_grad = False
        print("✓ Vision Encoder (CLIP):       FROZEN")
    else:
        print("✓ Vision Encoder (CLIP):       Trainable")
    
    # Cross Attentionの凍結
    if config.freeze_cross_attention:
        for param in model.cross_attention.parameters():
            param.requires_grad = False
        print("✓ Cross Attention:             FROZEN")
    else:
        print("✓ Cross Attention:             Trainable")
    
    # Classifierは常に学習可能
    print("✓ Classifier (Linear):         Trainable (always)")
    
    # 学習可能なパラメータ数を表示
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"\n{'='*60}")
    print("Parameter Statistics:")
    print(f"{'='*60}")
    print(f"Trainable parameters:  {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Frozen parameters:     {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
    print(f"Total parameters:      {total_params:,}")
    print(f"{'='*60}\n")


def decode_predictions(
    predicted_ids: torch.Tensor, 
    tokenizer,
    blank_token_id: int = 0
) -> List[str]:
    """
    CTCの予測結果をテキストにデコード
    
    Args:
        predicted_ids: 予測されたトークンID [batch, seq_len]
        tokenizer: トークナイザー
        blank_token_id: CTCのblankトークンID（通常は0）
        
    Returns:
        デコードされたテキストのリスト
    """
    decoded_texts = []
    
    for pred_ids_seq in predicted_ids:
        # CTCデコーディング：
        # 1. blank tokenを除去
        # 2. 連続する同じトークンを1つに統合（collapse）
        pred_tokens = []
        prev_token = None
        
        for token_id in pred_ids_seq.tolist():
            # blank tokenはスキップ
            if token_id == blank_token_id:
                prev_token = None  # blankが出たらprev_tokenをリセット
                continue
            
            # 連続する同じトークンは1つだけ保持
            if token_id != prev_token:
                pred_tokens.append(token_id)
            
            prev_token = token_id
        
        # トークンをテキストに変換
        decoded_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        decoded_texts.append(decoded_text)
    
    return decoded_texts


def compute_ctc_loss(
    logits: torch.Tensor,
    texts: List[str],
    tokenizer,
    wav_lengths: torch.Tensor
) -> torch.Tensor:
    """
    CTC損失を計算
    
    Args:
        logits: モデル出力 [B, T, vocab_size]
        texts: ターゲットテキスト [B]
        tokenizer: トークナイザー
        wav_lengths: 各音声の長さ [B]
    
    Returns:
        loss: CTC損失
    """
    batch_size = logits.size(0)
    device = logits.device
    
    # ターゲットのトークン化
    # CTCではスペース区切りの文字列が必要
    target_ids = []
    target_lengths = []
    
    for text in texts:
        # テキストをトークンIDに変換
        tokens = tokenizer.encode(text, add_special_tokens=False)
        target_ids.extend(tokens)
        target_lengths.append(len(tokens))
    
    # Tensorに変換
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    
    # 入力長を計算（logitsのシーケンス長）
    # Wav2Vec2は音声を約50倍圧縮するため、実際のシーケンス長はモデル出力から取得
    input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=device)
    
    # CTCLossの計算
    # logitsを [T, B, vocab_size] に転置
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = log_probs.transpose(0, 1)  # [T, B, vocab_size]
    
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    try:
        loss = ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
    except RuntimeError as e:
        print(f"\n[Warning] CTC Loss calculation error: {e}")
        print(f"  Input lengths: {input_lengths.tolist()}")
        print(f"  Target lengths: {target_lengths.tolist()}")
        # エラー時は大きな損失を返す
        loss = torch.tensor(1e6, device=device, requires_grad=True)
    
    return loss


def train_one_epoch(
    model: VisionConditionedASR,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    device: torch.device,
    epoch: int,
    config: TrainingConfig
):
    """
    1エポック分の学習
    
    Args:
        model: モデル
        dataloader: データローダー
        optimizer: オプティマイザー
        tokenizer: トークナイザー
        device: デバイス
        epoch: 現在のエポック番号
        config: 学習設定
    
    Returns:
        avg_loss: 平均損失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{config.num_epochs} - Training")
    print(f"{'='*60}")
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # データをデバイスに移動（wav_lengthsのみ）
            wav_lengths = batch["wav_lengths"].to(device)
            
            # Forward pass
            logits = model(batch)  # [B, T, vocab_size]
            
            # NaN/Infチェック
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n🚨 CRITICAL: Logits contain NaN or Inf at batch {batch_idx}!")
                print(f"  Skipping this batch...")
                continue
            
            # CTC損失の計算
            loss = compute_ctc_loss(logits, batch["text"], tokenizer, wav_lengths)
            
            # NaN/Infチェック
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n🚨 CRITICAL: Loss is NaN or Inf at batch {batch_idx}!")
                print(f"  Skipping this batch...")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
            
            # 損失の累積
            total_loss += loss.item()
            
            # ログ出力
            if (batch_idx + 1) % config.log_step == 0 or (batch_idx + 1) == num_batches:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  [{epoch+1}][{batch_idx+1}/{num_batches}] "
                      f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
            
            # メモリクリア
            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"\n[Error] Exception at batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1} Training Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"{'='*60}\n")
    
    return avg_loss


def validate(
    model: VisionConditionedASR,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
    num_examples: int = 3
):
    """
    検証
    
    Args:
        model: モデル
        dataloader: 検証データローダー
        tokenizer: トークナイザー
        device: デバイス
        epoch: 現在のエポック番号
        config: 学習設定
        num_examples: 表示する予測例の数
    
    Returns:
        avg_loss: 平均損失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_references = []
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{config.num_epochs} - Validation")
    print(f"{'='*60}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                # データをデバイスに移動
                wav_lengths = batch["wav_lengths"].to(device)
                
                # Forward pass
                logits = model(batch)
                
                # 損失計算
                loss = compute_ctc_loss(logits, batch["text"], tokenizer, wav_lengths)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
                
                # 予測のデコード
                predicted_ids = torch.argmax(logits, dim=-1)
                pred_texts = decode_predictions(predicted_ids, tokenizer, blank_token_id=0)
                
                # 結果の保存
                all_predictions.extend(pred_texts)
                all_references.extend(batch["text"])
                
            except Exception as e:
                print(f"\n[Error] Exception at validation batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 予測例を表示
    print(f"\n{'='*60}")
    print("Prediction Examples:")
    print(f"{'='*60}")
    
    for i in range(min(num_examples, len(all_predictions))):
        print(f"\nExample {i+1}:")
        print(f"  Reference:  {all_references[i][:80]}")
        print(f"  Prediction: {all_predictions[i][:80]}")
    
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1} Validation Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Total samples: {len(all_predictions)}")
    print(f"{'='*60}\n")
    
    return avg_loss


def save_checkpoint(
    model: VisionConditionedASR,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    config: TrainingConfig
):
    """
    チェックポイントを保存
    
    Args:
        model: モデル
        optimizer: オプティマイザー
        epoch: エポック番号
        train_loss: 訓練損失
        val_loss: 検証損失
        config: 学習設定
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f"checkpoint_epoch_{epoch+1}.pt"
    )
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }, checkpoint_path)
    
    print(f"[Checkpoint] Saved to {checkpoint_path}")


def main():
    """メイン学習関数"""
    # 設定の初期化
    config = TrainingConfig()
    
    # デバイスの設定
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Num epochs: {config.num_epochs}")
    print(f"{'='*60}\n")
    
    # トークナイザーの初期化
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # モデルの初期化
    print("[Setup] Initializing model...")
    model = VisionConditionedASR(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        device=device
    ).to(device)
    
    # 層の凍結
    freeze_layers(model, config)
    
    # データローダーの作成
    print("[Setup] Creating dataloaders...")
    train_loader = create_dataloader(
        json_path=config.train_json,
        audio_dir=config.audio_dir,
        image_dir=config.image_dir,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        max_audio_length=config.max_audio_length,
        validate_files=config.validate_files
    )
    
    val_loader = create_dataloader(
        json_path=config.val_json,
        audio_dir=config.audio_dir,
        image_dir=config.image_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        max_audio_length=config.max_audio_length,
        validate_files=config.validate_files
    )
    
    # オプティマイザーの設定
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学習ループ
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # 訓練
        train_loss = train_one_epoch(
            model, train_loader, optimizer, tokenizer, device, epoch, config
        )
        
        # 検証
        if (epoch + 1) % config.validate_epoch == 0:
            val_loss = validate(
                model, val_loader, tokenizer, device, epoch, config
            )
            
            # ベストモデルの保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\n✨ New best validation loss: {best_val_loss:.4f}")
                save_checkpoint(
                    model, optimizer, epoch, train_loss, val_loss, config
                )
        
        # 定期的なチェックポイント保存
        if (epoch + 1) % config.save_epoch == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss, 
                val_loss if 'val_loss' in locals() else 0.0, config
            )
    
    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()