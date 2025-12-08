from typing import List
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class TextAdapter(nn.Module):
    """
    Text adapter using CLIP text encoder for text-image alignment.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", embed_dim: int = 64):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        # Project to shared embedding dimension (64)
        # CLIP base output is 512, large is 768
        self.projection = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        
    def encode(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Encode a list of text strings into embeddings.
        Args:
            texts: List of strings
            device: torch device
        Returns:
            embeddings: (B, embed_dim)
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        
        # Use pooled output (EOS token)
        pooled_output = outputs.pooler_output  # (B, hidden_size)
        
        # Project to embedding dimension
        embeddings = self.projection(pooled_output) # (B, embed_dim)
        
        return embeddings
