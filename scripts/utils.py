from pydub import AudioSegment
import numpy as np

def load_m4a_file(audio_path):
    """Load M4A file using pydub"""
    try:
        
        audio_segment = AudioSegment.from_file(audio_path, format="m4a")
        
        audio_data = audio_segment.get_array_of_samples()
        audio = np.array(audio_data).astype(np.float32)
        
        if audio_segment.channels == 2:
            audio = audio.reshape((-1, 2)).mean(axis=1)
        
        audio = audio / 32768.0
        
        return audio, audio_segment.frame_rate
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None, None