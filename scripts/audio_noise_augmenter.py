import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
import random
import os
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class NoiseAugmentation:
    """Class to handle various noise augmentation techniques"""
    
    def __init__(self, 
                 noise_factor: float = 0.1,
                 background_noise_dir: Optional[str] = "data/musan",
                 augmentation_prob: float = 0.5):
        
        self.noise_factor = noise_factor
        self.augmentation_prob = augmentation_prob
        self.background_noise_files = []
        
        
        if background_noise_dir and os.path.exists(background_noise_dir):
            self._load_musan_files(background_noise_dir)
            logger.info(f"Loaded {len(self.background_noise_files)} background noise files from MUSAN dataset")
    
    def _load_musan_files(self, musan_dir: str):
        """Recursively load audio files from MUSAN dataset structure"""
        audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
        
        for root, dirs, files in os.walk(musan_dir):
            for file in files:
                if file.lower().endswith(audio_extensions):
                    file_path = os.path.join(root, file)
                    self.background_noise_files.append(file_path)
                    
        noise_files = [f for f in self.background_noise_files if '/noise/' in f]
        music_files = [f for f in self.background_noise_files if '/music/' in f]
        
        logger.info(f"MUSAN dataset breakdown - Noise: {len(noise_files)}, Music: {len(music_files)}")
    
    def add_white_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add white noise to waveform"""
        noise = torch.randn_like(waveform) * self.noise_factor
        return waveform + noise
    
    def add_background_noise(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Add background noise from noise files"""
        if not self.background_noise_files:
            return waveform
        
        noise_file = random.choice(self.background_noise_files)
        
        try:
            
            noise_waveform, noise_sr = torchaudio.load(noise_file)
            
            if noise_sr != sample_rate:
                resampler = T.Resample(orig_freq=noise_sr, new_freq=sample_rate)
                noise_waveform = resampler(noise_waveform)
            
            
            if noise_waveform.shape[0] > 1:
                noise_waveform = noise_waveform.mean(dim=0, keepdim=True)
            
            
            if noise_waveform.shape[1] > waveform.shape[1]:
                
                start_idx = random.randint(0, noise_waveform.shape[1] - waveform.shape[1])
                noise_waveform = noise_waveform[:, start_idx:start_idx + waveform.shape[1]]
            elif noise_waveform.shape[1] < waveform.shape[1]:
                
                repeat_factor = (waveform.shape[1] // noise_waveform.shape[1]) + 1
                noise_waveform = noise_waveform.repeat(1, repeat_factor)
                noise_waveform = noise_waveform[:, :waveform.shape[1]]
            
            
            noise_scale = random.uniform(0.05, self.noise_factor)
            return waveform + noise_waveform * noise_scale
            
        except Exception as e:
            logger.warning(f"Error loading noise file {noise_file}: {e}")
            return waveform
    
    def time_shift(self, waveform: torch.Tensor, max_shift: float = 0.1) -> torch.Tensor:
        """Apply random time shift"""
        shift_samples = int(random.uniform(-max_shift, max_shift) * waveform.shape[1])
        
        if shift_samples > 0:
            
            padded = torch.cat([torch.zeros(waveform.shape[0], shift_samples), waveform], dim=1)
            return padded[:, :waveform.shape[1]]
        elif shift_samples < 0:
            
            shifted = waveform[:, -shift_samples:]
            padded = torch.cat([shifted, torch.zeros(waveform.shape[0], -shift_samples)], dim=1)
            return padded
        else:
            return waveform
    
    def speed_perturbation(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply speed perturbation (0.9x to 1.1x speed)"""
        speed_factor = random.uniform(0.9, 1.1)
        
        
        effects = [
            ["speed", str(speed_factor)],
            ["rate", str(sample_rate)]
        ]
        
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate, effects
            )
            return augmented
        except:
            
            new_length = int(waveform.shape[1] / speed_factor)
            if new_length > 0:
                resampled = torch.nn.functional.interpolate(
                    waveform.unsqueeze(0), 
                    size=new_length, 
                    mode='linear'
                ).squeeze(0)
                
                
                if resampled.shape[1] > waveform.shape[1]:
                    return resampled[:, :waveform.shape[1]]
                else:
                    padding = waveform.shape[1] - resampled.shape[1]
                    return torch.cat([resampled, torch.zeros(waveform.shape[0], padding)], dim=1)
            return waveform
    
    def volume_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random volume scaling"""
        volume_factor = random.uniform(0.7, 1.4)
        return waveform * volume_factor
    
    def sudden_volume_change(self, waveform: torch.Tensor, 
                         min_changes: int = 1, 
                         max_changes: int = 3,
                         min_segment_duration: float = 0.3,
                         volume_range: Tuple[float, float] = (0.2, 2.0)) -> torch.Tensor:
        """
        Apply sudden volume changes at random points in the audio
        
        Args:
            waveform: Input audio tensor [channels, samples]
            min_changes: Minimum number of volume changes
            max_changes: Maximum number of volume changes
            min_segment_duration: Minimum duration for each segment (in fraction of total length)
            volume_range: Range of volume multipliers (min, max)
        
        Returns:
            Audio with sudden volume changes applied
        """
        if waveform.shape[1] < 1000:  
            return waveform
        
        augmented = waveform.clone()
        total_samples = waveform.shape[1]
        min_segment_samples = int(min_segment_duration * total_samples)
        
        num_changes = random.randint(min_changes, max_changes)
        
        change_points = []
        for _ in range(num_changes):
            
            valid_points = []
            for potential_point in range(min_segment_samples, 
                                    total_samples - min_segment_samples):
                
                too_close = any(abs(potential_point - cp) < min_segment_samples 
                            for cp in change_points)
                if not too_close:
                    valid_points.append(potential_point)
            
            if valid_points:
                change_points.append(random.choice(valid_points))
        
        
        change_points.sort()
        change_points = [0] + change_points + [total_samples]
        
        
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = change_points[i + 1]
            
            
            volume_factor = random.uniform(volume_range[0], volume_range[1])
            
            
            augmented[:, start_idx:end_idx] *= volume_factor
        
        
        transition_samples = random.randint(0, 100)  
        
        if transition_samples > 0:
            for change_point in change_points[1:-1]:  
                
                start_transition = max(0, change_point - transition_samples // 2)
                end_transition = min(total_samples, change_point + transition_samples // 2)
                
                if end_transition > start_transition:
                    
                    vol_before = augmented[:, start_transition].abs().mean() if start_transition > 0 else 0
                    vol_after = augmented[:, end_transition].abs().mean() if end_transition < total_samples else 0
                    
                    
                    fade_length = end_transition - start_transition
                    fade_curve = torch.linspace(0, 1, fade_length)
                    
                    for ch in range(augmented.shape[0]):
                        segment = augmented[ch, start_transition:end_transition]
                        
                        augmented[ch, start_transition:end_transition] = (
                            segment * fade_curve + 
                            segment * (1 - fade_curve)
                        )
        
        return augmented
    
    def apply_augmentation(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply random augmentations"""
        if random.random() > self.augmentation_prob:
            return waveform
        
        augmented = waveform.clone()
        
        
        augmentations = []
        
        if random.random() < 0.3:  
            augmentations.append('white_noise')
        
        if random.random() < 0.2 and self.background_noise_files:  
            augmentations.append('background_noise')
        
        if random.random() < 0.2:  
            augmentations.append('time_shift')
        
        if random.random() < 0.1:  
            augmentations.append('speed_perturbation')
        
        if random.random() < 0.3:  
            augmentations.append('volume_perturbation')
        
        if random.random() < 0.2:  
            augmentations.append('sudden_volume_change')
        
        for aug in augmentations:
            if aug == 'white_noise':
                augmented = self.add_white_noise(augmented)
            elif aug == 'background_noise':
                augmented = self.add_background_noise(augmented, sample_rate)
            elif aug == 'time_shift':
                augmented = self.time_shift(augmented)
            elif aug == 'speed_perturbation':
                augmented = self.speed_perturbation(augmented, sample_rate)
            elif aug == 'volume_perturbation':
                augmented = self.volume_perturbation(augmented)
            elif aug == 'sudden_volume_change':
                augmented = self.sudden_volume_change(augmented)
        
        return augmented
