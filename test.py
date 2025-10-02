import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import numpy as np

# ローカルファイルのインポート
from dataloader import SpokenCOCODataset, spokenCOCO_collate
from model import VisionConditionedASR

def calculate_cer(reference, hypothesis):
    """文字誤り率(CER)を計算"""
    dist_matrix = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        dist_matrix[i][0] = i
    for j in range(len(hypothesis) + 1):
        dist_matrix[0][j] = j

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            dist_matrix[i][j] = min(dist_matrix[i-1][j] + 1,
                                    dist_matrix[i][j-1] + 1,
                                    dist_matrix[i-1][j-1] + cost)
    
    if len(reference) == 0:
        return 0
    
    return dist_matrix[len(reference)][len(hypothesis)] / len(reference)

def decode_predictions(logits, tokenizer, pad_token_id, verbose=False):
    """CTCの予測結果をデコード"""
    predicted_ids = torch.argmax(logits, dim=-1)
    
    if verbose:
        print(f"\n[DEBUG] Logits shape: {logits.shape}")
        print(f"[DEBUG] Predicted IDs shape: {predicted_ids.shape}")
        print(f"[DEBUG] First 10 predicted IDs: {predicted_ids[0, :10].tolist()}")
        print(f"[DEBUG] Unique predicted IDs: {torch.unique(predicted_ids[0]).tolist()}")
        print(f"[DEBUG] Pad token ID: {pad_token_id}")
    
    decoded_texts = []
    for i in range(predicted_ids.size(0)):
        pred_tokens = []
        prev_token = None
        for t in predicted_ids[i].tolist():
            if t != pad_token_id and t != prev_token:
                pred_tokens.append(t)
            prev_token = t
        
        if verbose:
            print(f"[DEBUG] Sample {i} - Filtered tokens: {pred_tokens[:20]}")
            print(f"[DEBUG] Sample {i} - Number of unique tokens: {len(pred_tokens)}")
        
        pred_str = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        decoded_texts.append(pred_str)
        
        if verbose:
            print(f"[DEBUG] Sample {i} - Decoded text: '{pred_str}'")
    
    return decoded_texts

def run_demo(model_path, num_samples=5, random_seed=42, device='cuda:0', verbose=True):
    """
    学習済みモデルを使ってデモを実行
    
    Args:
        model_path: 学習済みモデルのパス
        num_samples: デモで使用するサンプル数
        random_seed: ランダムシード
        device: 使用するデバイス
        verbose: デバッグ情報を表示するかどうか
    """
    # デバイスの設定
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データセットのパス設定
    audio_dir_val = '../Datasets/SpokenCOCO/'
    image_dir_val = '../Datasets/stair_captions/images/'
    val_json_path = '../Datasets/SpokenCOCO/SpokenCOCO_val.json'
    
    # データセットの読み込み
    print("Loading validation dataset...")
    val_dataset = SpokenCOCODataset(
        json_path=val_json_path,
        audio_dir=audio_dir_val,
        image_dir=image_dir_val
    )
    
    # ランダムにサンプルを選択
    random.seed(random_seed)
    sample_indices = random.sample(range(len(val_dataset)), min(num_samples, len(val_dataset)))
    
    print(f"\nSelected {len(sample_indices)} samples for demo")
    print(f"Indices: {sample_indices}")
    
    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    if verbose:
        print(f"\n[DEBUG] Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"[DEBUG] Pad token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        print(f"[DEBUG] UNK token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    
    # モデルの読み込み
    print(f"\nLoading model from: {model_path}")
    from transformers import AutoModelForCTC
    temp_model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    vocab_size = temp_model.config.vocab_size
    
    if verbose:
        print(f"[DEBUG] Model vocab size: {vocab_size}")
    
    model = VisionConditionedASR(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # デモの実行
    print("\n" + "="*80)
    print("DEMO RESULTS")
    print("="*80)
    
    total_cer = 0
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(sample_indices):
            print(f"\n--- Sample {idx + 1}/{len(sample_indices)} (Dataset Index: {sample_idx}) ---")
            
            # データの取得
            sample = val_dataset[sample_idx]
            
            # バッチ化
            batch = spokenCOCO_collate([sample])
            
            # データをデバイスに移動
            data = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(device)
                else:
                    data[key] = value
            
            if verbose and idx == 0:
                print(f"[DEBUG] Audio shape: {len(batch['wav'][0])}")
                print(f"[DEBUG] Image type: {type(batch['image'][0])}")
                print(f"[DEBUG] Text: {batch['text'][0]}")
            
            # 推論
            logits = model(data)
            
            if verbose and idx == 0:
                print(f"[DEBUG] Output logits shape: {logits.shape}")
                print(f"[DEBUG] Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
                print(f"[DEBUG] Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
                
                # 最初のタイムステップの確率分布を確認
                probs = torch.softmax(logits[0, 0], dim=-1)
                top5_probs, top5_indices = torch.topk(probs, 5)
                print(f"[DEBUG] First timestep - Top 5 token probabilities:")
                for prob, idx_token in zip(top5_probs, top5_indices):
                    print(f"  Token {idx_token.item()}: {prob.item():.4f}")
            
            # デコード
            predictions = decode_predictions(logits, tokenizer, tokenizer.pad_token_id, 
                                           verbose=(verbose and idx == 0))
            
            # 正解テキスト
            reference = sample["text"]
            prediction = predictions[0]
            
            # CER計算
            cer = calculate_cer(
                reference.lower().replace(" ", ""), 
                prediction.lower().replace(" ", "")
            )
            total_cer += cer
            
            # 結果の表示
            print(f"\nReference: {reference}")
            print(f"Prediction: {prediction}")
            if len(prediction) == 0:
                print("  [WARNING] Prediction is empty!")
            print(f"CER: {cer:.4f}")
            
            # 音声と画像のパス情報
            if verbose:
                print(f"\nAudio path: {val_dataset.data[sample_idx]['wav']}")
                print(f"Image path: {val_dataset.data[sample_idx]['image']}")
    
    # 平均CERの計算
    avg_cer = total_cer / len(sample_indices)
    
    print("\n" + "="*80)
    print(f"Average CER: {avg_cer:.4f}")
    print("="*80)

def run_full_evaluation(model_path, device='cuda:0', batch_size=4):
    """
    全validationデータセットで評価
    
    Args:
        model_path: 学習済みモデルのパス
        device: 使用するデバイス
        batch_size: バッチサイズ
    """
    # デバイスの設定
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データセットのパス設定
    audio_dir_val = '../Datasets/SpokenCOCO/'
    image_dir_val = '../Datasets/stair_captions/images/'
    val_json_path = '../Datasets/SpokenCOCO/SpokenCOCO_val.json'
    
    # データセットの読み込み
    print("Loading validation dataset...")
    val_dataset = SpokenCOCODataset(
        json_path=val_json_path,
        audio_dir=audio_dir_val,
        image_dir=image_dir_val
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=spokenCOCO_collate
    )
    
    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # モデルの読み込み
    print(f"\nLoading model from: {model_path}")
    from transformers import AutoModelForCTC
    temp_model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    vocab_size = temp_model.config.vocab_size
    
    model = VisionConditionedASR(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
    
    # 評価ループ
    print("\nEvaluating on full validation set...")
    total_cer = 0
    total_samples = 0
    empty_predictions = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluation")):
            try:
                # データをデバイスに移動
                data = {}
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(device)
                    else:
                        data[key] = value
                
                # 推論
                logits = model(data)
                
                # デコード
                predictions = decode_predictions(logits, tokenizer, tokenizer.pad_token_id)
                
                # CER計算
                for i in range(len(data["text"])):
                    reference = data["text"][i]
                    prediction = predictions[i]
                    
                    if len(prediction) == 0:
                        empty_predictions += 1
                    
                    cer = calculate_cer(
                        reference.lower().replace(" ", ""), 
                        prediction.lower().replace(" ", "")
                    )
                    total_cer += cer
                    total_samples += 1
                
                # メモリ解放
                if batch_idx % 50 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    avg_cer = total_cer / total_samples
    
    print("\n" + "="*80)
    print(f"Full Validation Results")
    print(f"Total samples: {total_samples}")
    print(f"Empty predictions: {empty_predictions} ({empty_predictions/total_samples*100:.1f}%)")
    print(f"Average CER: {avg_cer:.4f}")
    print("="*80)

def main():
    # ==================== 検証設定 ====================
    # モデルのパス
    model_path = './model/vision_conditioned_asr_epoch_1.pth'
    
    # モード選択: 'demo' または 'full'
    # 'demo': ランダムサンプルで推論結果を詳細表示
    # 'full': 全validationデータセットで評価
    mode = 'demo'
    
    # デモモード設定
    num_samples = 5        # デモで使用するサンプル数
    random_seed = 42       # ランダムシード
    verbose = True         # デバッグ情報を表示
    
    # 共通設定
    device = 'cpu'         # 使用するデバイス ('cuda:0', 'cuda:1', 'cpu' など)
    batch_size = 4         # 完全評価モードのバッチサイズ
    # =================================================
    
    # モデルファイルの存在確認
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if mode == 'demo':
        run_demo(
            model_path=model_path,
            num_samples=num_samples,
            random_seed=random_seed,
            device=device,
            verbose=verbose
        )
    elif mode == 'full':
        run_full_evaluation(
            model_path=model_path,
            device=device,
            batch_size=batch_size
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'demo' or 'full'.")

if __name__ == "__main__":
    main()