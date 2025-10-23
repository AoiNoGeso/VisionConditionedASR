from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC, CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
from PIL import Image
import torchaudio
import numpy as np
import os

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
        self.processor = AutoProcessor.from_pretrained(model_name, force_download=True)
        self.model = AutoModelForCTC.from_pretrained(model_name, force_download=True)
        self.model.freeze_feature_encoder()

        if hasattr(self.model, 'wav2vec2') and hasattr(self.model.wav2vec2, 'masked_spec_embed'):
            if self.model.wav2vec2.masked_spec_embed is not None:
                nn.init.uniform_(self.model.wav2vec2.masked_spec_embed.data, a=-0.01, b=0.01)
        
        self.vocab_size = self.model.config.vocab_size
        self._device = device
        
    def forward(self, data):
        wav = data["wav"]
        
        if not wav or len(wav) == 0:
            raise ValueError("Empty audio input received")
        
        for i, w in enumerate(wav):
            if len(w) == 0:
                raise ValueError(f"Audio sample {i} has zero length")
        
        if self._device is not None:
            device = self._device
        else:
            device = next(self.model.parameters()).device
        
        processed = self.processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = processed.input_values.to(device)
        attention_mask = processed.attention_mask.to(device) if hasattr(processed, 'attention_mask') else None
        
        if torch.isnan(input_values).any() or torch.isinf(input_values).any():
            raise RuntimeError("Input values contain NaN or Inf after processing")
        
        audio_outputs = self.model.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        
        audio_features = audio_outputs.last_hidden_state
        
        if torch.isnan(audio_features).any():
            raise RuntimeError("Audio features contain NaN values")
        if torch.isinf(audio_features).any():
            raise RuntimeError("Audio features contain Inf values")
        
        return audio_features


class VisionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", device=None):
        super().__init__()
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        
        self._device = device
        
    def forward(self, data):
        images = data["image"]
        
        if not images or len(images) == 0:
            raise ValueError("Empty image input received")
        
        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                raise TypeError(f"Image {i} is not a PIL.Image object: {type(img)}")
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError(f"Image {i} has invalid size: {img.size}")
        
        if self._device is not None:
            device = self._device
        else:
            device = next(self.model.parameters()).device
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        image_features = self.model.get_image_features(**inputs)
        
        if torch.isnan(image_features).any():
            raise RuntimeError("Image features contain NaN values")
        if torch.isinf(image_features).any():
            raise RuntimeError("Image features contain Inf values")
        
        return image_features


class CrossAttention(nn.Module):
    def __init__(self, audio_dim=768, vision_dim=512, hidden_dim=256, num_heads=4):
        super().__init__()
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.audio_query = nn.Linear(audio_dim, hidden_dim)
        self.vision_key = nn.Linear(vision_dim, hidden_dim)
        self.vision_value = nn.Linear(vision_dim, hidden_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
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
        vision_features = vision_features.unsqueeze(1)
        
        audio_q = self.audio_query(audio_features)
        vision_k = self.vision_key(vision_features)
        vision_v = self.vision_value(vision_features)
        
        attn_output, _ = self.multihead_attn(
            query=audio_q,
            key=vision_k,
            value=vision_v,
            need_weights=True
        )
        
        projected_output = self.output_proj(attn_output)
        output = self.layer_norm(audio_features + self.dropout(projected_output))
        
        return output


class VisionConditionedASR(nn.Module):
    def __init__(self, vocab_size=None, hidden_dim=256, num_heads=4, device=None):
        super().__init__()
        self._device = device
        
        self.audio_encoder = AudioEncoder(device=device)
        self.vision_encoder = VisionEncoder(device=device)
        
        if vocab_size is None:
            vocab_size = self.audio_encoder.vocab_size
        
        self.vocab_size = vocab_size
        
        self.cross_attention = CrossAttention(
            audio_dim=768,
            vision_dim=512,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        self.classifier = nn.Linear(768, vocab_size)
        
    def forward(self, data):
        audio_features = self.audio_encoder(data)
        vision_features = self.vision_encoder(data)
        enhanced_audio = self.cross_attention(audio_features, vision_features)
        
        del audio_features, vision_features
        
        output_logits = self.classifier(enhanced_audio)
        
        return output_logits