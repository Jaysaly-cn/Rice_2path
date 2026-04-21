import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from model.transfer_lci import LCI_ConvNeXt
from utils.datasets import get_dataloaders

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels)

def main():
    TRAINED_MODEL_PATH = "./checkpoints/best_model_sota.pth"

    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    _, _, test_loader = get_dataloaders(config)
    
    real_num_classes = len(test_loader.dataset.classes)
    if real_num_classes != config['num_classes']:
        config['num_classes'] = real_num_classes

    model = LCI_ConvNeXt(
        num_classes=config['num_classes'], 
        model_name=config['backbone'],
        pretrained_path=None,
        dropout_rate=config['dropout_rate']
    )
    
    if os.path.exists(TRAINED_MODEL_PATH):
        print(f"Loading weights from: {TRAINED_MODEL_PATH}")
        checkpoint = torch.load(TRAINED_MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint)
    else:
        print(f"Error: Model weights not found at {TRAINED_MODEL_PATH}")
        return

    model = model.to(device)

    preds, labels = test(model, test_loader, device)
    
    acc = accuracy_score(labels, preds)
    
    precision_macro = precision_score(labels, preds, average='macro')
    recall_macro = recall_score(labels, preds, average='macro')
    f1_macro = f1_score(labels, preds, average='macro')
    
    precision_micro = precision_score(labels, preds, average='micro')
    recall_micro = recall_score(labels, preds, average='micro')
    f1_micro = f1_score(labels, preds, average='micro')
    
    print("\n" + "="*40)
    print(f"Evaluation Results")
    print("="*40)
    print(f"Overall Accuracy:  {acc:.4f}")
    print("-" * 40)
    print(f"Macro Precision:   {precision_macro:.4f}")
    print(f"Macro Recall:      {recall_macro:.4f}")
    print(f"Macro F1 Score:    {f1_macro:.4f}")
    print("-" * 40)
    print(f"Micro Precision:   {precision_micro:.4f}")
    print(f"Micro Recall:      {recall_micro:.4f}")
    print(f"Micro F1 Score:    {f1_micro:.4f}")
    print("="*40)
    
    print("\nDetailed Classification Report:")
    print(classification_report(labels, preds, digits=4))

if __name__ == '__main__':
    main()