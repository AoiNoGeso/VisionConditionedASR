import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np
from tqdm import tqdm
from pyctcdecode import build_ctcdecoder
import jiwer
from safetensors.torch import load_file

from model import VisionConditionedASR
from purewav2vec2_train import PureWav2Vec2ASR
from dataloader import create_dataloader
from train import TrainingConfig


@dataclass
class TestConfig:
    """評価設定"""
    # チェックポイント設定
    checkpoint_dir: str = "../checkpoints/fp16_model/epoch_5"  # エポックディレクトリを指定
    model_type: str = "vision"  # "pure" or "vision" - モデルタイプを指定
    
    # データセット設定
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    # モデル設定（VisionConditionedASR用、念のため）
    vocab_size: Optional[int] = None
    hidden_dim: int = 256
    num_heads: int = 4
    
    # データローダー設定
    batch_size: int = 16
    num_workers: int = 4
    max_audio_length: float = 10.0
    validate_files: bool = True
    
    # デコーディング設定
    use_beam_search: bool = True
    beam_width: int = 10
    
    # デバイス設定
    device: str = "cuda:1"
    
    # 結果保存設定
    save_results: bool = True
    results_dir: str = "../results/VASR/fp16_model"


class CTCDecoder:
    """pyctcdecodeを使用したCTCデコーダー"""
    
    def __init__(self, tokenizer, use_beam_search: bool = True, beam_width: int = 100):
        """
        Args:
            tokenizer: Hugging Face tokenizer
            use_beam_search: ビームサーチを使用するか
            beam_width: ビームの幅
        """
        self.tokenizer = tokenizer
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width
        
        # 語彙リストの作成
        vocab_list = []
        for i in range(tokenizer.vocab_size):
            token = tokenizer.convert_ids_to_tokens(i)
            # トークンが文字列でない場合は空文字列に変換
            if token is None:
                token = ""
            vocab_list.append(token)
        
        # pyctcdecodeのデコーダーを構築（言語モデルなし）
        self.decoder = build_ctcdecoder(
            labels=vocab_list,
            kenlm_model_path=None  # 言語モデルなし
        )
        
        print(f"[CTCDecoder] Initialized")
        print(f"  Vocabulary size: {len(vocab_list)}")
        print(f"  Beam search: {use_beam_search}")
        if use_beam_search:
            print(f"  Beam width: {beam_width}")
    
    def decode(self, logits: torch.Tensor) -> List[str]:
        """
        ログits配列をテキストにデコード
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
        
        Returns:
            デコードされたテキストのリスト
        """
        batch_size = logits.size(0)
        results = []
        
        # バッチ内の各サンプルをデコード
        for i in range(batch_size):
            logits_i = logits[i].cpu().numpy()  # [seq_len, vocab_size]
            
            if self.use_beam_search:
                # ビームサーチデコーディング
                text = self.decoder.decode(
                    logits_i,
                    beam_width=self.beam_width
                )
            else:
                # 貪欲法デコーディング
                text = self.decoder.decode(logits_i, beam_width=1)
            
            results.append(text)
        
        return results


def compute_wer(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
    WER（Word Error Rate）を計算
    
    Args:
        references: 正解テキストのリスト
        hypotheses: 予測テキストのリスト
    
    Returns:
        WER統計情報の辞書
    """
    output = jiwer.process_words(references, hypotheses)
    
    # アライメントから統計を集計
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_hits = 0
    
    for alignment in output.alignments:
        for op in alignment:
            if op.type == 'substitute':
                total_substitutions += 1
            elif op.type == 'delete':
                total_deletions += 1
            elif op.type == 'insert':
                total_insertions += 1
            elif op.type == 'equal':
                total_hits += 1
    
    return {
        'wer': output.wer * 100,  # パーセンテージに変換
        'mer': output.mer * 100,
        'wil': output.wil * 100,
        'substitutions': total_substitutions,
        'deletions': total_deletions,
        'insertions': total_insertions,
        'hits': total_hits
    }


def load_checkpoint(
    checkpoint_dir: str, 
    model: nn.Module, 
    device: torch.device,
    model_type: str = "vision"
):
    """
    チェックポイントからモデルを読み込む（safetensors形式対応）
    
    Args:
        checkpoint_dir: チェックポイントディレクトリのパス (例: ../checkpoints/epoch_4)
        model: モデルインスタンス
        device: デバイス
        model_type: "pure" or "vision" - モデルのタイプ
    
    Returns:
        epoch: 学習済みエポック数
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # エポック番号を取得
    epoch_num = os.path.basename(checkpoint_dir).split('_')[-1]
    
    # モデルファイルのパス
    model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch_num}.safetensors")
    state_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}_state.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"State file not found: {state_path}")
    
    print(f"\n[Loading] Loading checkpoint from: {checkpoint_dir}")
    print(f"[Loading] Model type: {model_type}")
    
    # モデルの重みをロード（safetensors）
    state_dict = load_file(model_path, device=str(device))
    model.load_state_dict(state_dict)
    
    # 学習状態をロード（.pt）
    checkpoint_state = torch.load(state_path, map_location=device, weights_only=False)
    
    # チェックポイント情報を表示
    epoch = checkpoint_state.get('epoch', -1)
    train_loss = checkpoint_state.get('train_loss', 0.0)
    val_loss = checkpoint_state.get('val_loss', 0.0)
    
    print(f"[Loading] Checkpoint loaded successfully")
    print(f"  Epoch: {epoch + 1}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    
    return epoch + 1


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    decoder: CTCDecoder,
    device: torch.device,
    config: TestConfig
):
    """
    モデルを評価してWERを計算
    
    Args:
        model: 評価するモデル（VisionConditionedASR or PureWav2Vec2ASR）
        dataloader: 検証データローダー
        decoder: CTCデコーダー
        device: デバイス
        config: 評価設定
    
    Returns:
        results: 評価結果の辞書
    """
    model.eval()
    
    all_references = []
    all_hypotheses = []
    all_samples = []  # 詳細な結果を保存
    
    print(f"\n{'='*60}")
    print("Starting Evaluation")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                # Forward pass
                logits = model(batch)  # [B, T, vocab_size]
                
                # デコーディング
                hypotheses = decoder.decode(logits)
                references = batch["text"]
                
                # 結果を保存
                all_references.extend(references)
                all_hypotheses.extend(hypotheses)
                
                # 詳細な結果を保存（最初の100サンプルのみ）
                if len(all_samples) < 100:
                    for ref, hyp in zip(references, hypotheses):
                        all_samples.append({
                            'reference': ref,
                            'hypothesis': hyp
                        })
                
            except Exception as e:
                print(f"\n[Error] Exception at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # WERの計算
    print(f"\n{'='*60}")
    print("Computing WER...")
    print(f"{'='*60}")
    
    wer_metrics = compute_wer(all_references, all_hypotheses)
    
    # 結果の表示
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_references)}")
    print(f"\nWord Error Rate (WER): {wer_metrics['wer']:.2f}%")
    print(f"Match Error Rate (MER): {wer_metrics['mer']:.2f}%")
    print(f"Word Information Lost (WIL): {wer_metrics['wil']:.2f}%")
    print(f"\nError Breakdown:")
    print(f"  Substitutions: {wer_metrics['substitutions']}")
    print(f"  Deletions:     {wer_metrics['deletions']}")
    print(f"  Insertions:    {wer_metrics['insertions']}")
    print(f"  Hits:          {wer_metrics['hits']}")
    print(f"{'='*60}\n")
    
    # サンプル結果の表示
    print(f"{'='*60}")
    print("Sample Predictions (first 5):")
    print(f"{'='*60}")
    for i, sample in enumerate(all_samples[:5]):
        print(f"\nSample {i+1}:")
        print(f"  Reference:  {sample['reference']}")
        print(f"  Hypothesis: {sample['hypothesis']}")
    print(f"{'='*60}\n")
    
    return {
        'wer_metrics': wer_metrics,
        'num_samples': len(all_references),
        'references': all_references,
        'hypotheses': all_hypotheses,
        'samples': all_samples
    }


def save_results(results: Dict, config: TestConfig, checkpoint_epoch: int, model_type: str):
    """
    評価結果を保存
    
    Args:
        results: 評価結果
        config: 評価設定
        checkpoint_epoch: チェックポイントのエポック数
        model_type: "pure" or "vision" - モデルのタイプ
    """
    os.makedirs(config.results_dir, exist_ok=True)
    
    # 結果ファイルのパス（モデルタイプを含める）
    results_file = os.path.join(
        config.results_dir,
        f"wer_results_{model_type}_epoch_{checkpoint_epoch}.txt"
    )
    
    # テキストファイルに保存
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"WER Evaluation Results ({model_type.upper()} Model)\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Checkpoint: {config.checkpoint_dir}\n")
        f.write(f"Epoch: {checkpoint_epoch}\n")
        f.write(f"Dataset: {config.val_json}\n")
        f.write(f"Beam Search: {config.use_beam_search}\n")
        if config.use_beam_search:
            f.write(f"Beam Width: {config.beam_width}\n")
        f.write(f"\nTotal samples: {results['num_samples']}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Metrics:\n")
        f.write("="*60 + "\n")
        f.write(f"WER: {results['wer_metrics']['wer']:.2f}%\n")
        f.write(f"MER: {results['wer_metrics']['mer']:.2f}%\n")
        f.write(f"WIL: {results['wer_metrics']['wil']:.2f}%\n")
        f.write(f"\nError Breakdown:\n")
        f.write(f"  Substitutions: {results['wer_metrics']['substitutions']}\n")
        f.write(f"  Deletions:     {results['wer_metrics']['deletions']}\n")
        f.write(f"  Insertions:    {results['wer_metrics']['insertions']}\n")
        f.write(f"  Hits:          {results['wer_metrics']['hits']}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Sample Predictions (first 20):\n")
        f.write("="*60 + "\n\n")
        
        for i, sample in enumerate(results['samples'][:20]):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  REF: {sample['reference']}\n")
            f.write(f"  HYP: {sample['hypothesis']}\n\n")
    
    print(f"[Results] Saved to: {results_file}\n")
    
    # 詳細な結果をCSVで保存
    csv_file = os.path.join(
        config.results_dir,
        f"predictions_{model_type}_epoch_{checkpoint_epoch}.csv"
    )
    
    import csv
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Reference', 'Hypothesis'])
        for i, (ref, hyp) in enumerate(zip(results['references'], results['hypotheses'])):
            writer.writerow([i+1, ref, hyp])
    
    print(f"[Results] Predictions saved to: {csv_file}\n")


def main():
    """メイン評価関数"""
    # 設定の初期化
    config = TestConfig()
    
    # デバイスの設定
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("Test Configuration")
    print(f"{'='*60}")
    print(f"Model Type: {config.model_type}")
    print(f"Device: {device}")
    print(f"Checkpoint: {config.checkpoint_dir}")
    print(f"Batch size: {config.batch_size}")
    print(f"Beam search: {config.use_beam_search}")
    if config.use_beam_search:
        print(f"Beam width: {config.beam_width}")
    print(f"{'='*60}\n")
    
    # トークナイザーの初期化
    print("[Setup] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # モデルの初期化（タイプに応じて）
    print("[Setup] Initializing model...")
    if config.model_type == "pure":
        model = PureWav2Vec2ASR(device=device).to(device)
        print("[Model] Using PureWav2Vec2ASR")
    elif config.model_type == "vision":
        model = VisionConditionedASR(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            device=device
        ).to(device)
        print("[Model] Using VisionConditionedASR")
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}. Must be 'pure' or 'vision'")
    
    # チェックポイントのロード
    checkpoint_epoch = load_checkpoint(
        config.checkpoint_dir, 
        model, 
        device,
        model_type=config.model_type
    )
    
    # デコーダーの初期化
    print("\n[Setup] Initializing decoder...")
    decoder = CTCDecoder(
        tokenizer=tokenizer,
        use_beam_search=config.use_beam_search,
        beam_width=config.beam_width
    )
    
    # データローダーの作成
    print("\n[Setup] Creating dataloader...")
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
    
    # 評価の実行
    results = evaluate(
        model=model,
        dataloader=val_loader,
        decoder=decoder,
        device=device,
        config=config
    )
    
    # 結果の保存
    if config.save_results:
        save_results(results, config, checkpoint_epoch, config.model_type)
    
    print("="*60)
    print("Evaluation Completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()