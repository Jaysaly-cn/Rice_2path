import os
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import random
from timm.data import Mixup
from timm.utils import ModelEmaV2

from model.transfer_lci import LCI_ConvNeXt
from utils.losses import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from utils.datasets import get_dataloaders, calculate_class_weights

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_one_epoch(model, ema_model, loader, criterion, optimizer, device, epoch, mixup_fn):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", unit="batch")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if ema_model is not None:
            ema_model.update(model)
        
        running_loss += loss.item()
        pbar.set_postfix({'Loss': running_loss/len(loader)})
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def save_checkpoint(state, is_best, checkpoint_dir):
    last_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
    torch.save(state, last_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model_ema.pth')
        torch.save(state['state_dict_ema'], best_path)

def main():
    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader = get_dataloaders(config)
    
    real_num_classes = len(train_loader.dataset.classes)
    if real_num_classes != config['num_classes']:
        print(f"Auto-adjusted num_classes: {real_num_classes}")
        config['num_classes'] = real_num_classes

    class_weights = calculate_class_weights(train_loader.dataset, device)

    print("Activating Mixup & CutMix...")
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=config['label_smoothing'],
        num_classes=config['num_classes']
    )

    model = LCI_ConvNeXt(
        num_classes=config['num_classes'], 
        model_name=config['backbone'], 
        pretrained_path=config['pretrained_path'],
        dropout_rate=config['dropout_rate']
    )
    model = model.to(device)
    
    print("Initializing EMA (Exponential Moving Average)...")
    model_ema = ModelEmaV2(model, decay=0.9998)

    train_criterion = SoftTargetCrossEntropy()
    val_criterion = LabelSmoothingCrossEntropy(eps=config['label_smoothing'], weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                            weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])

    start_epoch = 1
    best_acc = 0.0
    early_stop_counter = 0

    resume_path = config.get('resume_path', '')
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        model.load_state_dict(checkpoint['state_dict'])
        if 'state_dict_ema' in checkpoint and model_ema is not None:
             model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
        
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resume successful. Start Epoch: {start_epoch}, Best Acc: {best_acc:.2f}%")
    else:
        if resume_path:
            print(f"Warning: Resume path {resume_path} not found. Starting from scratch.")

    os.makedirs(config['checkpoints_dir'], exist_ok=True)

    for epoch in range(start_epoch, config['epochs'] + 1):
        train_loss = train_one_epoch(model, model_ema, train_loader, train_criterion, optimizer, device, epoch, mixup_fn)
        
        val_loss, val_acc = validate(model_ema.module, val_loader, val_criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss (EMA): {val_loss:.4f} Acc (EMA): {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            early_stop_counter = 0
            print(f"New Best EMA Model! Accuracy: {best_acc:.2f}%")
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

        save_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'state_dict_ema': model_ema.module.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        save_checkpoint(save_dict, is_best, config['checkpoints_dir'])

    print(f"Training Finished. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    main()