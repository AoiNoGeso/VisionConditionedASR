from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC, CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import torchaudio

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    
    def forward(self, data):
        """
        data["wav"]: バッチの場合 List[np.ndarray], 単一の場合 np.ndarray
        """
        wav = data["wav"]
        
        # GPU使用時のデバイス移動
        device = next(self.model.parameters()).device
        
        input_values = self.processor(
            wav,
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).input_values.to(device)
        
        audio_outputs = self.model.wav2vec2(input_values)
        audio_features = audio_outputs.last_hidden_state # [1, time, hidden_dim]
        
        return audio_features
    
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        
    def forward(self, data):
        """
        data["image"]: Tensor[B, 3, H, W] または Tensor[1, 3, H, W]
        """
        images = data["image"]
                             
        # CLIP processorで処理
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        
        # GPU使用時のデバイス移動
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # CLIP vision encoderで特徴抽出
        image_features = self.model.get_image_features(**inputs)
        
        return image_features

class CrossAttention(nn.Module):
    def __init__(self, audio_dim=768, vision_dim=512, hidden_dim=256, num_heads=2):
        super().__init__()
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.audio_query = nn.Linear(audio_dim, hidden_dim)
        self.audio_key = nn.Linear(audio_dim, hidden_dim)
        self.audio_value = nn.Linear(audio_dim, hidden_dim)
        
        self.vision_key = nn.Linear(vision_dim, hidden_dim)
        self.vision_value = nn.Linear(vision_dim, hidden_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
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
        
        # Project features
        audio_q = self.audio_query(audio_features)
        audio_k = self.audio_key(audio_features)
        audio_v = self.audio_value(audio_features)
        
        vision_k = self.vision_key(vision_features)   # [B, 1, hidden_dim]
        vision_v = self.vision_value(vision_features) # [B, 1, hidden_dim]
        
        # Concatenate for cross-attention
        keys = torch.cat([audio_k, vision_k], dim=1)
        values = torch.cat([audio_v, vision_v], dim=1)
        
        # Apply multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            query=audio_q,
            key=keys,
            value=values
        )
        
        projected_output = self.output_proj(attn_output)
        
        output = self.layer_norm(audio_features + self.dropout(projected_output))
        
        return output

class VisionConditionedASR(nn.Module):
    def __init__(self, vocab_size=32, hidden_dim=256, num_heads=2):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.vision_encoder = VisionEncoder()
        self.cross_attention = CrossAttention(
            audio_dim=768,
            vision_dim=512,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        self.classifier = nn.Linear(768, vocab_size)
        
    def forward(self, data):
        # 音声特徴抽出: [B, seq_len, 768]
        audio_logits = self.audio_encoder(data)      
        
        # 画像特徴抽出: [B, 512]
        vision_features = self.vision_encoder(data)  
        
        # クロスアテンション: [B, seq_len, 768]
        enhanced_audio = self.cross_attention(audio_logits, vision_features)
        
        # 分類: [B, seq_len, vocab_size]
        output_logits = self.classifier(enhanced_audio)  
        
        return output_logits

def demo():
    print("デモを実行します")
    
    ae = AudioEncoder()
    ve = VisionEncoder()
    avsr = VisionConditionedASR()
    
    wav_path = "../Datasets/SpokenCOCO/wavs/val/0/m071506418gb9vo0w5xq3-3LUY3GC63Z0R9PYEETJGN5HO4UEP7B_325114_629297.wav"
    img_path = "../Datasets/stair_captions/images/val2014/COCO_val2014_000000325114.jpg"
    
    # 音声データ読み込み
    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    # [channels, samples] -> [1, samples] (モノラルに変換)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # 画像データ読み込み
    image = Image.open(img_path).convert('RGB')
    
    dataset = [
        {"wav": wav.numpy(),
         "text": "A URINAL IN A PUBLIC RESTROOM NEAR A WOODEN TABLE",
         "image": image}
    ]
    
    sample = dataset[0]
    
    audio_outputs = ae(data=sample)
    vision_outputs = ve(data=sample)
    asr_outputs = avsr(data=sample)
    
    print(f"audio outputs: {audio_outputs.shape}")
    print(f"vision outputs: {vision_outputs.shape}")
    print(f"ASR outputs: {asr_outputs.shape}")
    
    # デコーディング処理を追加
    print("\n--- デコーディング結果 ---")
    
    # 音声エンコーダーのtokenizerを取得
    tokenizer = avsr.audio_encoder.tokenizer
    
    # logitsから予測されたトークンIDを取得
    # asr_outputs: [batch, seq_len, vocab_size]
    predicted_ids = torch.argmax(asr_outputs, dim=-1)  # [batch, seq_len]
    
    # バッチ内の各サンプルをデコード
    for i, pred_ids in enumerate(predicted_ids):
        # トークンIDをテキストに変換
        # pred_ids: [seq_len]
        pred_ids = pred_ids.cpu().numpy()
        
        # tokenizerでデコード
        transcription = tokenizer.decode(pred_ids)
        
        print(f"サンプル {i}:")
        print(f"  予測テキスト: {transcription}")
        print(f"  正解テキスト: {sample['text']}")
        
        # 重複した文字を削除したバージョンも表示（CTC出力の特性上）
        # CTCでは同じ文字が連続することがあるため
        unique_ids = []
        prev_id = -1
        for token_id in pred_ids:
            if token_id != prev_id:
                unique_ids.append(token_id)
                prev_id = token_id
        
        transcription_unique = tokenizer.decode(unique_ids)
        print(f"  予測テキスト(重複除去): {transcription_unique}")
        print()
    
if __name__ == "__main__":
    demo()