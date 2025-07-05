from wav2vec_encoder import run_encoder_from_tensor, Wav2VecEncoder
import torch
import torchaudio
from classifier_model import AudioClassifier
from time import time
from pydub import AudioSegment
import numpy as np
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wav2vec_model = Wav2VecEncoder(model_name="facebook/wav2vec2-xls-r-300m", device=device)
threshold = 0.5

vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
vad_model = vad_model.to(device)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def detect_speech_segments(audio_path):
    
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.sample_width == 2:  
        samples = samples / 32768.0
    
    
    wav_tensor = torch.from_numpy(samples).to(device)
    
    
    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        vad_model,
        threshold=0.5,
        sampling_rate=16000,
        return_seconds=True,
        min_speech_duration_ms=250,        
        min_silence_duration_ms=300,
        speech_pad_ms=50
    )
    
    merged_segments = []
    
    for segment in speech_timestamps:
        start = segment['start']
        end = segment['end']
        
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        segment_tensor = wav_tensor[start_sample:end_sample]
        
        merged_segments.append(segment_tensor)
    
    if len(merged_segments) == 0:
        return None, None, None
    
    return torch.cat(merged_segments, dim=0), audio.duration_seconds, audio.frame_rate

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)    
    input_dim = checkpoint['input_dim']
    hidden_dims = checkpoint['hidden_dims']
    model = AudioClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model

def run_inference(model, audio_path):
    speech_segments, audio_duration, audio_sample_rate = detect_speech_segments(audio_path)
    
    if speech_segments is None:
        return "No speech detected"
    
    print(f"audio duration: {audio_duration} seconds")
    print(f"audio sample rate: {audio_sample_rate}")
    
    feature = run_encoder_from_tensor(
        audio_tensor=speech_segments,
        model=wav2vec_model,
        sample_rate=audio_sample_rate
    )
    
    feature = feature.mean(dim=1)  
    
    if device == "cuda":
        feature = feature.cpu()

    feature = feature.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(feature)  
        
        probs = torch.softmax(output, dim=1)
        
        prob_male = probs[0, 1].item()
        
        if prob_male > threshold:
            return "male"
        else:
            return "female"

def main(audio_path, model_path):
    
    model = load_model(model_path)
    
    if model is None:
        raise FileNotFoundError("Could not find model file in any of the expected locations")
    
    start_time = time()
    result = run_inference(model, audio_path)
    end_time = time()
    
    print(f"Time taken for inference: {round(end_time - start_time, 2)} seconds")
    print(f"device: {device}")
    print("="*20)
    print(f"Gender: {result}")
    print("="*20)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--audio_path", type=str, required=True)
    args.add_argument("--model_path", type=str, default="exp/models/best_model.pt")
    args = args.parse_args()
    
    main(args.audio_path, args.model_path)
