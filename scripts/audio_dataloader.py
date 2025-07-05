import torch
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
from typing import List, Tuple, Optional
import logging
import tempfile
import os
from audio_noise_augmenter import NoiseAugmentation
from wav2vec_encoder import run_encoder_from_tensor
from audio_dataset import AugmentedAudioDataset
from audio_noise_augmenter import NoiseAugmentation

logger = logging.getLogger(__name__)

class Wav2VecAudioDataLoader:
    """DataLoader for wav2vec processing with variable-length audio"""
    
    def __init__(self,
                 audio_files: List[str],
                 labels: List[int],
                 wav2vec_model,  
                 batch_size: int = 32,
                 sample_rate: int = 16000,
                 augmentation: Optional[NoiseAugmentation] = None,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 use_file_based: bool = False):
        
        self.wav2vec_model = wav2vec_model
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.use_file_based = use_file_based
        
        
        self.dataset = AugmentedAudioDataset(
            audio_files=audio_files,
            labels=labels,
            sample_rate=sample_rate,
            augmentation=augmentation
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=False
        )
    
    def _collate_fn(self, batch):
        """Custom collate function that processes audio through wav2vec using run_encoder"""
        waveforms = []
        labels = []
        
        for waveform, label in batch:
            waveforms.append(waveform)
            labels.append(label)
        
        
        features = []
            
        for waveform in waveforms:
            
            feature = run_encoder_from_tensor(
                audio_tensor=waveform,
                model=self.wav2vec_model,
                sample_rate=self.sample_rate
            )
            
            
            feature = feature.mean(dim=1)  
            features.append(feature.squeeze(0).cpu())
        
        
        features_tensor = torch.stack(features)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return features_tensor, labels_tensor
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_wav2vec_train_dataloader(audio_files: List[str],
                                   labels: List[int],
                                   wav2vec_model,
                                   batch_size: int = 32,
                                   sample_rate: int = 16000,
                                   noise_factor: float = 0.1,
                                   background_noise_dir: Optional[str] = None,
                                   augmentation_prob: float = 0.5,
                                   num_workers: int = 4,
                                   use_file_based: bool = False):
    """Create training dataloader with wav2vec processing"""    
    
    augmentation = NoiseAugmentation(
        noise_factor=noise_factor,
        background_noise_dir=background_noise_dir,
        augmentation_prob=augmentation_prob
    )
    
    return Wav2VecAudioDataLoader(
        audio_files=audio_files,
        labels=labels,
        wav2vec_model=wav2vec_model,
        batch_size=batch_size,
        sample_rate=sample_rate,
        augmentation=augmentation,
        shuffle=True,
        num_workers=num_workers,
        use_file_based=use_file_based
    )

def create_wav2vec_val_dataloader(audio_files: List[str],
                                 labels: List[int],
                                 wav2vec_model,
                                 batch_size: int = 32,
                                 sample_rate: int = 16000,
                                 num_workers: int = 4,
                                 use_file_based: bool = False):
    """Create validation dataloader with wav2vec processing (no augmentation)"""
    
    return Wav2VecAudioDataLoader(
        audio_files=audio_files,
        labels=labels,
        wav2vec_model=wav2vec_model,
        batch_size=batch_size,
        sample_rate=sample_rate,
        augmentation=None,  
        shuffle=False,
        num_workers=num_workers,
        use_file_based=use_file_based
    ) 