import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from typing import Optional, Union
import warnings
import os

class Wav2VecEncoder:
    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m", device: str = "cuda"):
        """
        Initialize Wav2Vec encoder.
        
        Args:
            model_name: Pretrained model name from HuggingFace
            device: Device to run the model on
        """
        self.device = device
        self.model_name = model_name
        
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        
        self.feature_dim = self.model.config.hidden_size
  
    def extract_features_from_tensor(self, audio_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from audio tensor using Wav2Vec.
        
        Args:
            audio_tensor: Audio tensor (1D or 2D with batch dimension)
            sample_rate: Sample rate of the audio
            
        Returns:
            Encoded features tensor
        """
        try:
            # Ensure audio is numpy array for feature extractor
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.cpu().numpy()
            else:
                audio_np = audio_tensor
            
            # If 2D, take first channel or flatten
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()
            
            # Process through feature extractor
            inputs = self.feature_extractor(
                audio_np, 
                sampling_rate=sample_rate, 
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state  
            
            return features
            
        except Exception as e:
            raise RuntimeError(f"Error processing audio tensor: {str(e)}")

def run_encoder_from_tensor(audio_tensor: torch.Tensor,
                           model: Wav2VecEncoder,
                           sample_rate: int = 16000) -> torch.Tensor:
    """
    Run wav2vec encoder on an audio tensor.
    More efficient alternative that doesn't require temporary files.
    
    Args:
        audio_tensor: Audio tensor
        model: Wav2VecEncoder instance
        sample_rate: Sample rate of the audio
    
    Returns:
        Encoded audio tensor
    """
    return model.extract_features_from_tensor(audio_tensor, sample_rate) 