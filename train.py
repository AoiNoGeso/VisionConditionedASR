import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCTC
from dataclasses import dataclass
import wandb
import os
import random
import numpy as np

from dataloader import SpokenCOCODataset, spokenCOCO_collate
from model import VisionConditionedASR

# ========================================
# 設定クラス
# ========================================
@dataclass
class TrainingConfig:
    """学習に関する設定パラメータ"""
    # シード値
    seed: int = 42
    
    # wandb設定
    use_wandb: bool = False  # wandbロギングを使用するかどうか
    wandb_project: str = "VisionConditionedASR"  # wandbプロジェクト名
    
    # 学習パラメータ
    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 16
    
    # 学習率設定（パラメータグループごと）
    learning_rate_audio_encoder: float = 1e-5   # Audio Encoder (pre trained)
    learning_rate_vision_encoder: float = 1e-5  # Vision Encoder (pre trained)
    learning_rate_cross_attention: float = 1e-5 # Cross Attention (new)
    learning_rate_classifier: float = 1e-5      # Classifier (new)
    
    # 学習率スケジューラー
    warmup_steps: int = 500                     # ウォームアップステップ数
    use_scheduler: bool = True                  # スケジューラーを使用するか
    
    # 勾配クリッピング
    max_grad_norm: float = 1.0                  # 勾配クリッピングの閾値
    
    # 層の凍結設定
    freeze_audio_encoder: bool = True         # Audio Encoderを凍結
    freeze_vision_encoder: bool = True         # Vision Encoderを凍結
    freeze_cross_attention: bool = False        # Cross Attentionを凍結
    
    # データセットパス（学習用）
    audio_dir_train: str = '../../Datasets/SpokenCOCO/'
    image_dir_train: str = '../../Datasets/stair_captions/images/'
    train_json_path: str = '../../Datasets/SpokenCOCO/SpokenCOCO_train_fixed.json'
    
    # データセットパス（評価用）
    audio_dir_val: str = '../../Datasets/SpokenCOCO/'
    image_dir_val: str = '../../Datasets/stair_captions/images/'
    val_json_path: str = '../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json'
    
    # モデル保存先
    model_save_dir: str = '../model/'


# ========================================
# ユーティリティ関数
# ========================================
def move_data_to_device(data, device):
    """
    バッチデータをGPU/CPUに転送
    
    Args:
        data: バッチデータ（辞書形式）
        device: 転送先デバイス
        
    Returns:
        デバイスに転送されたデータ
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def calculate_wer(reference, hypothesis):
    """
    WER (Word Error Rate) を計算
    
    単語レベルでの編集距離を用いて、音声認識の精度を評価します。
    値が小さいほど精度が高いことを示します。
    
    Args:
        reference: 正解テキスト
        hypothesis: 予測テキスト
    
    Returns:
        WER値（0.0〜1.0以上）
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    # 編集距離行列を初期化
    n_ref = len(ref_words)
    n_hyp = len(hyp_words)
    dist_matrix = [[0] * (n_hyp + 1) for _ in range(n_ref + 1)]
    
    # 境界条件
    for i in range(n_ref + 1):
        dist_matrix[i][0] = i
    for j in range(n_hyp + 1):
        dist_matrix[0][j] = j
    
    # 動的計画法で編集距離を計算
    for i in range(1, n_ref + 1):
        for j in range(1, n_hyp + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            
            dist_matrix[i][j] = min(
                dist_matrix[i-1][j] + 1,      # 削除
                dist_matrix[i][j-1] + 1,      # 挿入
                dist_matrix[i-1][j-1] + cost  # 置換
            )
    
    # WERを計算（空文字列の場合の処理を含む）
    if n_ref == 0:
        return 0 if n_hyp == 0 else 1
    
    return dist_matrix[n_ref][n_hyp] / n_ref


def set_seed(seed: int):
    """
    再現性のためにシード値を固定
    
    Args:
        seed: 固定するシード値
    """
    print(f"\n[Seed] Setting random seed to {seed}")
    
    # Python標準のrandomモジュール
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # マルチGPU対応
    
    # CuDNNの決定的動作を有効化（若干速度が低下する可能性あり）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("  ✓ Random seed has been set for reproducibility")
    print("  ✓ CuDNN deterministic mode enabled")


def decode_predictions(predicted_ids, tokenizer, pad_token_id):
    """
    CTCの予測結果をテキストにデコード
    
    Args:
        predicted_ids: 予測されたトークンID [batch, seq_len]
        tokenizer: トークナイザー
        pad_token_id: パディングトークンのID
        
    Returns:
        デコードされたテキストのリスト
    """
    decoded_texts = []
    
    for pred_ids_seq in predicted_ids:
        # CTCデコーディング：連続する同じトークンを除去
        pred_tokens = []
        prev_token = None
        
        for token_id in pred_ids_seq.tolist():
            if token_id != pad_token_id and token_id != prev_token:
                pred_tokens.append(token_id)
            prev_token = token_id
        
        # トークンをテキストに変換
        decoded_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        decoded_texts.append(decoded_text)
    
    return decoded_texts


# ========================================
# モデル設定関数
# ========================================
def freeze_layers(model, config):
    """
    指定された層を凍結
    
    Args:
        model: VisionConditionedASRモデル
        config: 学習設定
    """
    print("\n[Freeze] Layer freeze configuration:")
    
    # Audio Encoderの凍結
    if config.freeze_audio_encoder:
        for param in model.audio_encoder.parameters():
            param.requires_grad = False
        print("  ✓ Audio Encoder: FROZEN")
    else:
        print("  ✓ Audio Encoder: Trainable")
    
    # Vision Encoderの凍結
    if config.freeze_vision_encoder:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        print("  ✓ Vision Encoder: FROZEN")
    else:
        print("  ✓ Vision Encoder: Trainable")
    
    # Cross Attentionの凍結
    if config.freeze_cross_attention:
        for param in model.cross_attention.parameters():
            param.requires_grad = False
        print("  ✓ Cross Attention: FROZEN")
    else:
        print("  ✓ Cross Attention: Trainable")
    
    # Classifierは常に学習可能
    print("  ✓ Classifier: Trainable (always)")


def get_optimizer(model, config):
    """
    パラメータグループごとに異なる学習率を設定したオプティマイザーを作成
    
    Args:
        model: VisionConditionedASRモデル
        config: 学習設定
        
    Returns:
        オプティマイザー
    """
    param_groups = []
    
    # Audio Encoder
    if not config.freeze_audio_encoder:
        param_groups.append({
            'params': model.audio_encoder.parameters(),
            'lr': config.learning_rate_audio_encoder,
            'name': 'audio_encoder'
        })
    
    # Vision Encoder
    if not config.freeze_vision_encoder:
        param_groups.append({
            'params': model.vision_encoder.parameters(),
            'lr': config.learning_rate_vision_encoder,
            'name': 'vision_encoder'
        })
    
    # Cross Attention
    if not config.freeze_cross_attention:
        param_groups.append({
            'params': model.cross_attention.parameters(),
            'lr': config.learning_rate_cross_attention,
            'name': 'cross_attention'
        })
    
    # Classifier
    param_groups.append({
        'params': model.classifier.parameters(),
        'lr': config.learning_rate_classifier,
        'name': 'classifier'
    })
    
    optimizer = optim.AdamW(param_groups)
    
    # 学習率の情報を表示
    print("\n[Optimizer] Learning rate configuration:")
    for group in param_groups:
        print(f"  ✓ {group['name']}: lr={group['lr']:.2e}")
    
    return optimizer


def get_scheduler(optimizer, config, total_steps):
    """
    ウォームアップ + コサインアニーリングスケジューラーを作成
    
    Args:
        optimizer: オプティマイザー
        config: 学習設定
        total_steps: 総学習ステップ数
        
    Returns:
        スケジューラー（使用しない場合はNone）
    """
    if not config.use_scheduler:
        print("\n[Scheduler] No scheduler will be used")
        return None
    
    # ウォームアップスケジューラー
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=config.warmup_steps
    )
    
    # コサインアニーリングスケジューラー
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - config.warmup_steps,
        eta_min=1e-7
    )
    
    # スケジューラーの組み合わせ
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps]
    )
    
    print("\n[Scheduler] Learning rate scheduler:")
    print(f"  ✓ Warmup steps: {config.warmup_steps}")
    print(f"  ✓ Total steps: {total_steps}")
    print(f"  ✓ Type: Linear warmup + Cosine annealing")
    
    return scheduler


# ========================================
# 学習関数
# ========================================
def train_one_epoch(model, dataloader, optimizer, scheduler, ctc_loss, 
                   tokenizer, config, device, epoch, global_step):
    """
    1エポック分の学習を実行
    
    Args:
        model: 学習対象のモデル
        dataloader: 学習データローダー
        optimizer: オプティマイザー
        scheduler: 学習率スケジューラー（Noneの場合もあり）
        ctc_loss: CTC損失関数
        tokenizer: トークナイザー
        config: 学習設定
        device: 実行デバイス
        epoch: 現在のエポック番号
        global_step: グローバルステップ数
        
    Returns:
        (平均損失, 更新されたglobal_step)
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs} (Train)")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # データをデバイスに転送
            data = move_data_to_device(batch, device)
            
            # ラベルの準備
            labels = tokenizer(
                data["text"], 
                return_tensors="pt", 
                padding=True
            ).input_ids.to(device)
            label_lengths = torch.sum(labels != tokenizer.pad_token_id, dim=1)
            
            # 順伝播
            logits = model(data)
            
            # NaN or Inf check
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n🚨🚨 CRITICAL ERROR: Logits contain NaN or Inf!")
                print(f"  Max value: {logits.abs().max()}")
                print(f"  Min value: {logits.abs().min()}")
                raise RuntimeError("Model output (logits) is numerically unstable.")
            
            # CTC損失の計算
            log_probs = torch.log_softmax(logits, dim=2).transpose(0, 1)
            input_lengths = torch.full(
                (logits.size(0),), 
                logits.size(1), 
                dtype=torch.long
            ).to(device)
            
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            
            # 勾配累積のための正規化
            loss = loss / config.gradient_accumulation_steps
            
            # 逆伝播
            loss.backward()
            
            # 損失の記録
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # 進捗バーの更新（現在の学習率も表示）
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                loss=loss.item() * config.gradient_accumulation_steps,
                lr=f"{current_lr:.2e}"
            )
            
            # パラメータ更新（勾配累積ステップごと）
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # 勾配クリッピング
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config.max_grad_norm
                )
                
                # パラメータ更新
                optimizer.step()
                optimizer.zero_grad()
                
                # スケジューラーの更新
                if scheduler is not None:
                    scheduler.step()
                
                # wandbへのロギング
                if config.use_wandb and (global_step + 1) % 100 == 0:
                    log_dict = {
                        "train/batch_loss": loss.item() * config.gradient_accumulation_steps,
                        "train/learning_rate": current_lr
                    }
                    wandb.log(log_dict, step=global_step)
                
                global_step += 1
            
            # メモリ管理
            if batch_idx % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n[Error] Batch {batch_idx} failed: {e}")
            print(f"  Batch keys: {batch.keys()}")
            if "wav" in batch:
                print(f"  wav type: {type(batch['wav'])}")
            if "image" in batch:
                print(f"  image count: {len(batch['image'])}")
            raise e
    
    # エポック終了時の残り勾配を適用
    if len(dataloader) % config.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=config.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, global_step


def validate(model, dataloader, ctc_loss, tokenizer, device):
    """
    検証データでモデルを評価
    
    Args:
        model: 評価対象のモデル
        dataloader: 検証データローダー
        ctc_loss: CTC損失関数
        tokenizer: トークナイザー
        device: 実行デバイス
        
    Returns:
        (平均損失, 平均WER)
    """
    model.eval()
    total_loss = 0
    total_wer = 0
    
    print("\n[Validation] Starting...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            try:
                # データをデバイスに転送
                data = move_data_to_device(batch, device)
                
                # ラベルの準備
                labels = tokenizer(
                    data["text"], 
                    return_tensors="pt", 
                    padding=True
                ).input_ids.to(device)
                label_lengths = torch.sum(labels != tokenizer.pad_token_id, dim=1)
                
                # 順伝播
                logits = model(data)
                
                # 損失の計算
                log_probs = torch.log_softmax(logits, dim=2).transpose(0, 1)
                input_lengths = torch.full(
                    (logits.size(0),), 
                    logits.size(1), 
                    dtype=torch.long
                ).to(device)
                
                loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
                total_loss += loss.item()
                
                # WERの計算
                predicted_ids = torch.argmax(logits, dim=-1)
                pred_texts = decode_predictions(
                    predicted_ids, 
                    tokenizer, 
                    tokenizer.pad_token_id
                )
                
                for pred_text, ref_text in zip(pred_texts, data["text"]):
                    total_wer += calculate_wer(ref_text, pred_text)
                
                # メモリ管理
                if batch_idx % 50 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\n[Error] Validation batch {batch_idx} failed: {e}")
                continue
    
    avg_loss = total_loss / len(dataloader)
    avg_wer = total_wer / len(dataloader.dataset)
    
    return avg_loss, avg_wer


def save_model(model, epoch, config):
    """
    モデルを保存
    
    Args:
        model: 保存するモデル
        epoch: 現在のエポック番号
        config: 学習設定
    """
    save_path = os.path.join(
        config.model_save_dir, 
        f'vision_conditioned_asr_epoch_{epoch+1}.pth'
    )
    torch.save(model.state_dict(), save_path)
    print(f"[Save] Model saved to {save_path}")


# ========================================
# メイン関数
# ========================================
def main():
    """メイン学習ループ"""
    # 設定の読み込み
    config = TrainingConfig()
    
    # シード値の設定
    set_seed(config.seed)
    
    # デバイスの設定
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Using device: {device}")
    
    # モデル保存ディレクトリの作成
    os.makedirs(config.model_save_dir, exist_ok=True)
    
    # wandbの初期化
    if config.use_wandb:
        wandb.init(project=config.wandb_project, config=config)
        print(f"[Wandb] Logging enabled - Project: {config.wandb_project}")
    else:
        print("[Wandb] Logging disabled")
    
    # ========================================
    # データセットの準備
    # ========================================
    print("\n[Dataset] Loading training data...")
    train_dataset = SpokenCOCODataset(
        json_path=config.train_json_path,
        audio_dir=config.audio_dir_train,
        image_dir=config.image_dir_train
    )
    
    print("\n[Dataset] Loading validation data...")
    val_dataset = SpokenCOCODataset(
        json_path=config.val_json_path,
        audio_dir=config.audio_dir_val,
        image_dir=config.image_dir_val
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=spokenCOCO_collate
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=spokenCOCO_collate
    )
    
    # ========================================
    # モデル・最適化の準備
    # ========================================
    print("\n[Model] Initializing model and optimizer...")
    
    # トークナイザーと語彙サイズの取得
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    temp_model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    vocab_size = temp_model.config.vocab_size
    
    # モデルの初期化
    model = VisionConditionedASR(vocab_size=vocab_size).to(device)
    
    # 層の凍結設定
    freeze_layers(model, config)
    
    # オプティマイザーの作成（パラメータグループごとに異なる学習率）
    optimizer = get_optimizer(model, config)
    
    # 総ステップ数の計算
    steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.epochs
    
    # 学習率スケジューラーの作成
    scheduler = get_scheduler(optimizer, config, total_steps)
    
    # 損失関数
    ctc_loss = nn.CTCLoss(blank=tokenizer.pad_token_id)
    
    print(f"\n[Training] Training configuration:")
    print(f"  ✓ Total epochs: {config.epochs}")
    print(f"  ✓ Steps per epoch: {steps_per_epoch}")
    print(f"  ✓ Total steps: {total_steps}")
    print(f"  ✓ Gradient clipping: max_norm={config.max_grad_norm}")
    
    # ========================================
    # 学習ループ
    # ========================================
    print("\n[Training] Starting training loop...")
    global_step = 0
    
    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"{'='*60}")
        
        # 学習フェーズ
        avg_train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            ctc_loss=ctc_loss,
            tokenizer=tokenizer,
            config=config,
            device=device,
            epoch=epoch,
            global_step=global_step
        )
        
        # 評価フェーズ
        avg_val_loss, avg_wer = validate(
            model=model,
            dataloader=val_dataloader,
            ctc_loss=ctc_loss,
            tokenizer=tokenizer,
            device=device
        )
        
        # 結果のロギング
        if config.use_wandb:
            wandb.log({
                "train/epoch_loss": avg_train_loss,
                "val/loss": avg_val_loss,
                "val/wer": avg_wer,
                "epoch": epoch
            })
        
        # 結果の表示
        print(f"\n[Results] Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val WER:    {avg_wer:.4f}")
        
        # モデルの保存
        save_model(model, epoch, config)
    
    # ========================================
    # 最終モデルの保存
    # ========================================
    print("\n[Training] Training finished!")
    final_save_path = os.path.join(
        config.model_save_dir, 
        'vision_conditioned_asr_final.pth'
    )
    torch.save(model.state_dict(), final_save_path)
    print(f"[Save] Final model saved to {final_save_path}")
    
    # wandbの終了
    if config.use_wandb:
        wandb.finish()
        print("[Wandb] Logging session finished")


if __name__ == "__main__":
    main()