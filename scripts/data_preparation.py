import pandas as pd
import numpy as np
import os
import wave
from pydub import AudioSegment

def is_valid_audio_file(file_path):
    """
    Check if the audio file exists and is valid.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        bool: True if file exists and is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        _ = AudioSegment.from_file(file_path, format="m4a")
        return True
    except Exception as e:
        print(f"Error reading audio file {file_path}: {str(e)}")
        return False

def prepare_training_data(df, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, random_state=42):
    """
    Prepare training data from a DataFrame containing audio paths and labels.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'audio_filepath' and 'label' columns
        train_ratio (float): Ratio of data to use for training (default: 0.9)
        val_ratio (float): Ratio of data to use for validation (default: 0.05)
        test_ratio (float): Ratio of data to use for testing (default: 0.05)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (train_df, val_df, test_df) containing the split datasets
    """
    
    required_columns = ['audio_filepath', 'label']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    if not df['label'].isin([0, 1]).all():
        raise ValueError("Labels must be binary (0 or 1)")
    
    print("Validating audio files...")
    valid_rows = []
    invalid_files = []
    
    for idx, row in df.iterrows():
        if is_valid_audio_file(row['audio_filepath']):
            valid_rows.append(row)
        else:
            invalid_files.append(row['audio_filepath'])
    
    if invalid_files:
        print(f"\nSkipping {len(invalid_files)} invalid audio files:")
        for file in invalid_files:
            print(f"- {file}")

    if not valid_rows:
        raise ValueError("No valid audio files found in the dataset!")

    print(f"\nFound {len(valid_rows)} valid audio files")
    

    df = pd.DataFrame(valid_rows)

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_samples = len(df)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    print(f"\nTotal samples: {n_samples}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print("\nLabel distribution:")
    for split_name, split_df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
        label_counts = split_df['label'].value_counts()
        print(f"\n{split_name} set:")
        for label, count in label_counts.items():
            print(f"Label {label}: {count} samples ({count/len(split_df)*100:.1f}%)")
    
    def save_labels_file(df, output_file):
        with open(output_file, 'w') as f:
            for _, row in df.iterrows():
                f.write(f"{row['audio_filepath']} {row['label']}\n")
    
    os.makedirs("dataset", exist_ok=True)
    
    save_labels_file(train_df, "data/train_labels.txt")
    save_labels_file(val_df, "data/val_labels.txt")
    save_labels_file(test_df, "data/test_labels.txt")


if __name__ == "__main__":

    df = pd.read_csv("data/VoxCeleb_gender_dataset.csv")  

    prepare_training_data(df)

    print("\nNow you can train with train_classifier.py:")
    print("Run the training script with:")
    print("bash train.sh") 
