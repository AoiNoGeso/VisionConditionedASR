import torch
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import torchaudio
import os

class SpokenCOCODataset(Dataset):
    def __init__(self, json_path, audio_dir=None, image_dir=None, sample_rate=16000):
        """
        json_path: jsonファイルのパス
        audio_dir: wavファイルのディレクトリ
        image_dir: 画像のディレクトリ
        sample_rate: 音声リサンプルレート
        """
        
        self.audio_dir = audio_dir if audio_dir is not None else ""
        self.image_dir = image_dir if image_dir is not None else ""
        self.sample_rate = sample_rate

        with open(json_path, 'r') as f:
            raw_data = json.load(f)
        
        # captions を展開して flat な list に（ファイル存在チェック付き）
        self.data = []
        total_items = 0
        missing_audio = 0
        missing_image = 0
        
        for item in raw_data["data"]:
            image_path = item["image"]
            full_image_path = os.path.join(self.image_dir, image_path)
            
            image_exists = os.path.exists(full_image_path)
            if not image_exists:
                missing_image += 1
                print(f"Missing image: {image_path}")
                continue
            
            for cap in item["captions"]:
                total_items += 1
                wav_path = cap["wav"]
                full_wav_path = os.path.join(self.audio_dir, wav_path)
                
                audio_exists = os.path.exists(full_wav_path)
                if not audio_exists:
                    missing_audio += 1
                    continue
                
                # 両方のファイルが存在する場合のみ追加
                self.data.append({
                    "image": full_image_path,
                    "wav": full_wav_path,
                    "text": cap["text"]
                })
        
        valid_items = len(self.data)
        # 統計情報を表示
        print(f"Dataset loading complete:")
        print(f"  Total items in JSON: {total_items}")
        print(f"  Missing image files: {missing_image}")
        print(f"  Missing audio files: {missing_audio}")
        print(f"  Valid items loaded: {valid_items}")
        print(f"  Success rate: {valid_items/total_items*100:.3f}%")
        
        if len(self.data) == 0:
            raise ValueError("No valid data found. Please check file paths and data availability.")
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            # --- 音声読み込み ---
            wav_path = item['wav']
            wav, sr = torchaudio.load(wav_path)  # (channels, T)

            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

            # モノラル化とチャンネル次元の除去
            if wav.size(0) > 1:  # ステレオの場合
                wav = wav.mean(dim=0, keepdim=True)  # (1, T)
            
            # 1次元にして返す
            wav = wav.squeeze(0)  # (T,)

            # --- 画像読み込み ---
            img_path = item['image']
            image = Image.open(img_path).convert("RGB")

            # --- テキスト ---
            text = item['text']

            return {"wav": wav, "image": image, "text": text}
        
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            print(f"  Audio path: {item['wav']}")
            print(f"  Image path: {item['image']}")
            # エラーが発生した場合、次のインデックスを試す
            return self.__getitem__((idx + 1) % len(self.data))

def spokenCOCO_collate(batch):
    """
    batch: list of dict {"wav": Tensor[T], "image": PIL.Image, "text": str}
    
    AutoProcessorに適合する形式でバッチを作成
    """
    # --- 音声: numpy配列のリストで返す ---
    waveforms = [b["wav"].numpy() for b in batch]
    
    # 長さ情報も保持（必要に応じて）
    lengths = [len(w) for w in waveforms]

    # --- 画像: PIL Imageのリストで返す（AutoProcessorがリストを期待） ---
    images = [b["image"] for b in batch]

    # --- テキスト: リストのまま ---
    texts = [b["text"] for b in batch]

    return {
        "wav": waveforms,      # List[np.ndarray] - AutoProcessorのaudio引数用
        "wav_lengths": torch.tensor(lengths),
        "image": images,       # List[PIL.Image] - AutoProcessorのimages引数用
        "text": texts          # List[str] - AutoProcessorのtext引数用
    }
