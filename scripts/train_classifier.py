import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import datetime
import multiprocessing

from audio_dataset import AugmentedAudioDataset
from audio_dataloader import create_wav2vec_train_dataloader, create_wav2vec_val_dataloader
from classifier_model import AudioClassifier
from wav2vec_encoder import Wav2VecEncoder


def setup_logger():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train audio classifier for answermachine detection")
    parser.add_argument("--train-labels", type=str, required=True, help="File containing training labels with full paths")
    parser.add_argument("--val-labels", type=str, required=True, help="File containing validation labels with full paths")
    parser.add_argument("--encoder-model", type=str, default="facebook/wav2vec2-xls-r-300m", help="HuggingFace model name (wav2vec)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--hidden-dims", type=str, default="1024,512,256", help="Comma-separated list of hidden layer dimensions")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Dimension of audio embeddings")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate of audio files")
    parser.add_argument("--tensorboard-dir", type=str, default="runs", help="Directory for TensorBoard logs")
    parser.add_argument("--noise-factor", type=float, default=0.1, help="Noise factor for augmentation (0.1 = 10% of signal)")
    parser.add_argument("--background-noise-dir", type=str, default=None, help="Directory containing background noise files")
    parser.add_argument("--augmentation-prob", type=float, default=0.5, help="Probability of applying augmentation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum audio length in samples")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume training from")
    return parser.parse_args()

def load_data(labels_file):
    """
    Load data from a label file containing full paths to audio files.
    
    Args:
        labels_file (str): Path to the label file
        
    Returns:
        tuple: (audio_files, labels) where audio_files contains full paths
    """
    audio_files = []
    labels = []
    
    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                file_path = parts[0]
                label = int(parts[1])
                if label not in [0, 1]:
                    raise ValueError(f"Label must be 0 (not answermachine) or 1 (answermachine), got {label} for {file_path}")
                if not os.path.exists(file_path):
                    raise ValueError(f"Audio file not found: {file_path}")
                audio_files.append(file_path)
                labels.append(label)
    
    return audio_files, labels

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        if len(batch) == 2:  
            inputs, targets = batch
            inputs = inputs.to(device)
        else:  
            inputs, lengths, targets = batch
            inputs = inputs.to(device)
        
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(model.predict_proba(inputs).detach().cpu().numpy())
        
        pbar.set_postfix({
            "loss": total_loss / (pbar.n + 1),
            "acc": 100. * correct / total
        })
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', pos_label=1
    )
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0
    
    metrics = {
        'loss': total_loss / len(loader),
        'acc': 100. * correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return metrics

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:  
                inputs, targets = batch
                inputs = inputs.to(device)
            else:  
                inputs, lengths, targets = batch
                inputs = inputs.to(device)
            
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(model.predict_proba(inputs).detach().cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', pos_label=1
    )
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0
    
    metrics = {
        'loss': total_loss / len(loader),
        'acc': 100. * correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return metrics

def load_checkpoint(checkpoint_path, model, optimizer, device):
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    start_epoch = checkpoint['epoch'] + 1
    
    
    model_config = {
        'input_dim': checkpoint.get('input_dim'),
        'hidden_dims': checkpoint.get('hidden_dims')
    }
    
    best_f1 = checkpoint.get('val_metrics', {}).get('f1', 0.0)
    
    return start_epoch, best_f1, model_config

def main():
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)
    
    args = parse_args()
    logger = setup_logger()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    tensorboard_dir = os.path.join(args.tensorboard_dir, f"run_{timestamp}")
    writer = SummaryWriter(tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  
        torch.backends.cudnn.deterministic = False  
        logger.info(f"Using device: {device}")
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        num_workers = 0
        logger.info("Setting num_workers=0 to avoid CUDA multiprocessing issues")
    else:
        device = torch.device("cpu")
        logger.info("CUDA is not available. Using CPU for training.")
        num_workers = args.num_workers
    
    logger.info("Loading training data...")
    train_audio_files, train_labels = load_data(args.train_labels)
    logger.info(f"Loaded {len(train_audio_files)} training files with classes: 0={train_labels.count(0)}, 1={train_labels.count(1)}")
    
    logger.info("Loading validation data...")
    val_audio_files, val_labels = load_data(args.val_labels)
    logger.info(f"Loaded {len(val_audio_files)} validation files with classes: 0={val_labels.count(0)}, 1={val_labels.count(1)}")
    
    encoder_model = None
        
    logger.info("Initializing wav2vec encoder model...")
    
    encoder_model = Wav2VecEncoder(
        model_name=args.encoder_model,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    args.embedding_dim = encoder_model.feature_dim
    logger.info(f"Wav2Vec model feature dimension: {encoder_model.feature_dim}")
    
    
    logger.info("Creating datasets...")
    logger.info("Creating dataloaders...")
    train_loader = create_wav2vec_train_dataloader(
        audio_files=train_audio_files,
        labels=train_labels,
        wav2vec_model=encoder_model,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        noise_factor=args.noise_factor,
        background_noise_dir=args.background_noise_dir,
        augmentation_prob=args.augmentation_prob,
        num_workers=num_workers
    )
    
    val_loader = create_wav2vec_val_dataloader(
        audio_files=val_audio_files,
        labels=val_labels,
        wav2vec_model=encoder_model,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        num_workers=num_workers
    )
    
    logger.info("Creating binary classifier model for answermachine detection...")
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]
    
    
    start_epoch = 0
    best_f1 = 0
    
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        
        if 'input_dim' in checkpoint and 'hidden_dims' in checkpoint:
            input_dim = checkpoint['input_dim']
            hidden_dims = checkpoint['hidden_dims']
            logger.info(f"Using model configuration from checkpoint: input_dim={input_dim}, hidden_dims={hidden_dims}")
        else:
            
            if encoder_model:
                input_dim = args.embedding_dim
            else:
                input_dim = 512
            logger.info(f"Using configuration from args: input_dim={input_dim}, hidden_dims={hidden_dims}")
    else:
        if encoder_model:
            input_dim = args.embedding_dim
        else:
            input_dim = 512
    
    model = AudioClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=2
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    
    if args.checkpoint:
        start_epoch, best_f1, model_config = load_checkpoint(args.checkpoint, model, optimizer, device)
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}, best F1: {best_f1:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%, " +
                   f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['acc'], epoch)
        writer.add_scalar('F1/train', train_metrics['f1'], epoch)
        writer.add_scalar('AUC/train', train_metrics['auc'], epoch)
        
        
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%, " +
                   f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['acc'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        
        
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims
        }
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            logger.info(f"New best F1 score: {val_metrics['f1']:.4f}")
            best_model_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
    
    
    final_model_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(checkpoint, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    
    writer.close()
    logger.info(f"Training completed. Best validation F1 score: {best_f1:.4f}")
    logger.info(f"To view TensorBoard logs, run: tensorboard --logdir={tensorboard_dir}")


if __name__ == "__main__":
    main()