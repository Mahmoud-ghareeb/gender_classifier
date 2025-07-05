import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import random
from typing import List, Tuple, Optional
import logging
from audio_noise_augmenter import NoiseAugmentation


logger = logging.getLogger(__name__)

class AugmentedAudioDataset(Dataset):
    """Audio dataset for wav2vec encoder - no max_length needed"""
    
    def __init__(self, 
                 audio_files: List[str], 
                 labels: List[int],
                 sample_rate: int = 16000,
                 augmentation: Optional[NoiseAugmentation] = None):
        """
        Args:
            audio_files: List of audio file paths
            labels: List of corresponding labels
            sample_rate: Target sample rate for wav2vec
            augmentation: NoiseAugmentation instance for data augmentation
        """
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate
        self.augmentation = augmentation
        
        # Cache resamplers for efficiency
        self.resampler = {}  
        
    def __len__(self):
        return len(self.audio_files)
    
    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file with error handling"""
        try:
            # Try torchaudio first
            try:
                waveform, sr = torchaudio.load(audio_path)
            except:
                # Fallback for unsupported formats (M4A, etc.)
                from pydub import AudioSegment
                import numpy as np
                
                audio_segment = AudioSegment.from_file(audio_path)
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                audio_data = audio_data / 32768.0  # Normalize
                waveform = torch.tensor(audio_data).unsqueeze(0)
                sr = audio_segment.frame_rate
            
            return waveform, sr
            
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return 1 second of silence as fallback
            return torch.zeros(1, self.sample_rate), self.sample_rate
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load audio
        waveform, original_sr = self._load_audio(audio_path)
        
        # Resample if necessary (wav2vec expects 16kHz)
        if original_sr != self.sample_rate:
            if original_sr not in self.resampler:
                self.resampler[original_sr] = T.Resample(
                    orig_freq=original_sr, 
                    new_freq=self.sample_rate
                )
            waveform = self.resampler[original_sr](waveform)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Apply augmentation during training
        if self.augmentation is not None:
            waveform = self.augmentation.apply_augmentation(waveform, self.sample_rate)
        
        # Return raw audio (variable length) - wav2vec will handle it
        return waveform.squeeze(0), torch.tensor(label, dtype=torch.long)
