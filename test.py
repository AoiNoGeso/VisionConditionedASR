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

from model import VisionConditionedASR
from dataloader import create_dataloader
from train import TrainingConfig


@dataclass
class TestConfig:
    """評価設定"""
    # チェックポイント設定
    checkpoint_path: str = "../checkpoints/checkpoint_epoch_4.pt"
    
    # データセット設定
    val_json: str = "../../Datasets/SpokenCOCO/SpokenCOCO_val_fixed.json"
    audio_dir: str = "../../Datasets/SpokenCOCO"
    image_dir: str = "../../Datasets/stair_captions/images"
    
    # モデル設定（チェックポイントから自動取得されるが、念のため）
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
    results_dir: str = "../results"


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
    # jiwerを使用してWERを計算
    wer = jiwer.wer(references, hypotheses)
    mer = jiwer.mer(references, hypotheses)
    wil = jiwer.wil(references, hypotheses)
    
    # 詳細な統計情報を取得
    measures = jiwer.process_words(references, hypotheses)
    
    return {
        'wer': wer * 100,  # パーセンテージに変換
        'mer': mer * 100,
        'wil': wil * 100,
        'substitutions': measures['substitutions'],
        'deletions': measures['deletions'],
        'insertions': measures['insertions'],
        'hits': measures['hits']
    }


def load_checkpoint(checkpoint_path: str, model: VisionConditionedASR, device: torch.device):
    """
    チェックポイントからモデルを読み込む
    
    Args:
        checkpoint_path: チェックポイントファイルのパス
        model: モデルインスタンス
        device: デバイス
    
    Returns:
        epoch: 学習済みエポック数
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n[Loading] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # モデルの重みをロード
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # チェックポイント情報を表示
    epoch = checkpoint.get('epoch', -1)
    train_loss = checkpoint.get('train_loss', 0.0)
    val_loss = checkpoint.get('val_loss', 0.0)
    
    print(f"[Loading] Checkpoint loaded successfully")
    print(f"  Epoch: {epoch + 1}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}")
    
    return epoch + 1


def evaluate(
    model: VisionConditionedASR,
    dataloader: DataLoader,
    decoder: CTCDecoder,
    device: torch.device,
    config: TestConfig
):
    """
    モデルを評価してWERを計算
    
    Args:
        model: 評価するモデル
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


def save_results(results: Dict, config: TestConfig, checkpoint_epoch: int):
    """
    評価結果を保存
    
    Args:
        results: 評価結果
        config: 評価設定
        checkpoint_epoch: チェックポイントのエポック数
    """
    os.makedirs(config.results_dir, exist_ok=True)
    
    # 結果ファイルのパス
    results_file = os.path.join(
        config.results_dir,
        f"wer_results_epoch_{checkpoint_epoch}.txt"
    )
    
    # テキストファイルに保存
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("WER Evaluation Results\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Checkpoint: {config.checkpoint_path}\n")
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
        f"predictions_epoch_{checkpoint_epoch}.csv"
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
    print(f"Device: {device}")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Batch size: {config.batch_size}")
    print(f"Beam search: {config.use_beam_search}")
    if config.use_beam_search:
        print(f"Beam width: {config.beam_width}")
    print(f"{'='*60}\n")
    
    # トークナイザーの初期化
    print("[Setup] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    
    # モデルの初期化
    print("[Setup] Initializing model...")
    model = VisionConditionedASR(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        device=device
    ).to(device)
    
    # チェックポイントのロード
    checkpoint_epoch = load_checkpoint(config.checkpoint_path, model, device)
    
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
        save_results(results, config, checkpoint_epoch)
    
    print("="*60)
    print("Evaluation Completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()