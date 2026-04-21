import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model.transfer_lci import LCI_ConvNeXt
from utils.datasets import RicePestDataset

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_model(config, device):
    model = LCI_ConvNeXt(
        num_classes=config['num_classes'], 
        model_name=config['backbone'],
        pretrained_path=None,
        dropout_rate=config['dropout_rate']
    )
    
    weight_path = os.path.join(config['checkpoints_dir'], 'best_model_ema.pth')
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f"Loaded weights from {weight_path}")
    else:
        raise FileNotFoundError(f"Weights not found at {weight_path}")
    
    model = model.to(device)
    model.eval()
    return model

def extract_data(model, loader, device):
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Extracting Data"):
            inputs = inputs.to(device)
            
            x = model.backbone(inputs)
            x = model.fusion(x)
            features = model.global_pool(x).flatten(1)
            outputs = model.head(features)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())
            
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), np.array(all_features)

def plot_confusion_matrix(y_true, y_pred, classes, output_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix saved to {output_path}")

def plot_tsne(features, labels, classes, output_path='tsne_projection.png'):
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 10))
    palette = sns.color_palette("hsv", len(classes))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=[classes[i] for i in labels], 
                    palette=palette, legend='full', s=60, alpha=0.8)
    plt.title('t-SNE Visualization of Feature Space')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"t-SNE plot saved to {output_path}")

def plot_roc(y_true, y_probs, classes, output_path='roc_curves.png'):
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ROC curves saved to {output_path}")

def main():
    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    test_dir = os.path.join(config['data_root'], 'test')
    test_ds = RicePestDataset(test_dir, is_training=False, image_size=config['input_size'])
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    config['num_classes'] = len(test_ds.classes)
    class_names = test_ds.classes
    
    model = get_model(config, device)
    
    preds, labels, probs, features = extract_data(model, test_loader, device)
    
    plot_confusion_matrix(labels, preds, class_names)
    plot_roc(labels, probs, class_names)
    plot_tsne(features, labels, class_names)

if __name__ == '__main__':
    main()