
import os
import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    
    weight_path = os.path.join(config['checkpoints_dir'], 'best_model_sota.pth')
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f"Loaded weights from {weight_path}")
    else:
        raise FileNotFoundError(f"Weights not found at {weight_path}")
    
    model = model.to(device)
    model.eval()
    return model

def extract_feature_batch(model, inputs):
    with torch.no_grad():
        x = model.backbone(inputs)
        x = model.fusion(x)
        x = model.global_pool(x)
        features = torch.flatten(x, 1)
        return F.normalize(features, p=2, dim=1)

def build_or_load_feature_bank(model, dataset, device, batch_size=64, num_workers=8, cache_file='feature_bank.npy'):
    if os.path.exists(cache_file):
        print(f"Loading feature bank from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True).item()
        return data['features'], data['paths'], data['labels']
    
    print("Building feature bank (Accelerated Mode)...")
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    features_list = []
    labels_list = []
    
    for inputs, labels in tqdm(loader, desc="Extracting Features"):
        inputs = inputs.to(device)
        feats = extract_feature_batch(model, inputs)
        features_list.append(feats.cpu().numpy())
        labels_list.extend(labels.numpy())
    
    processed_count = sum([f.shape[0] for f in features_list])
    all_samples = dataset.samples
    paths_list = [s[0] for s in all_samples[:processed_count]]
    
    features = np.concatenate(features_list, axis=0)
    
    data = {
        'features': features,
        'paths': paths_list,
        'labels': labels_list
    }
    np.save(cache_file, data)
    print(f"Feature bank built! Shape: {features.shape}")
    return features, paths_list, labels_list

def visualize_comparison(query_path, retrieved_path, query_class, retrieved_class, similarity, conf, output_path='retrieval_result.jpg'):
    img_query = cv2.imread(query_path)
    img_retrieved = cv2.imread(retrieved_path)
    
    if img_query is None or img_retrieved is None:
        print("Error reading images for visualization.")
        return

    img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
    img_retrieved = cv2.cvtColor(img_retrieved, cv2.COLOR_BGR2RGB)
    
    img_query = cv2.resize(img_query, (400, 400))
    img_retrieved = cv2.resize(img_retrieved, (400, 400))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img_query)
    axes[0].set_title(f"Query Image\nPred: {query_class}\nConf: {conf:.2%}", color='blue', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_retrieved)
    axes[1].set_title(f"Most Similar in DB\nLabel: {retrieved_class}\nSimilarity: {similarity:.4f}", color='green', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.show()
    print(f"Result saved to {output_path}")

def main():
    QUERY_IMAGE_PATH = "/data4/Agri/yukaijie/Rice_2path/data/test/1.jpg"

    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    train_dir = os.path.join(config['data_root'], 'train')
    train_ds = RicePestDataset(train_dir, is_training=False, image_size=config['input_size'])
    class_names = train_ds.classes
    config['num_classes'] = len(class_names)
    
    model = get_model(config, device)
    
    bank_features, bank_paths, bank_labels = build_or_load_feature_bank(
        model, train_ds, device, batch_size=64, num_workers=8
    )
    bank_features = torch.from_numpy(bank_features).to(device)

    if not os.path.exists(QUERY_IMAGE_PATH):
        QUERY_IMAGE_PATH = input("Please input query image path: ")

    img_transform = train_ds.transform_logic
    query_tensor = img_transform(QUERY_IMAGE_PATH).unsqueeze(0).to(device)
    
    query_feat = extract_feature_batch(model, query_tensor)
    
    with torch.no_grad():
        logits = model(query_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_conf, pred_idx = torch.max(probs, 1)
        pred_class_name = class_names[pred_idx.item()]
    
    similarity = torch.mm(query_feat, bank_features.t())
    best_sim, best_idx = torch.max(similarity, 1)
    
    best_idx = best_idx.item()
    best_sim = best_sim.item()
    
    retrieved_path = bank_paths[best_idx]
    retrieved_label_idx = bank_labels[best_idx]
    retrieved_class_name = class_names[retrieved_label_idx]
    
    print("-" * 50)
    print(f"Query Prediction: {pred_class_name} ({pred_conf.item():.2%})")
    print(f"Retrieved Image:  {retrieved_path}")
    print(f"Retrieved Class:  {retrieved_class_name}")
    print(f"Cosine Similarity: {best_sim:.4f}")
    print("-" * 50)
    
    visualize_comparison(
        QUERY_IMAGE_PATH, 
        retrieved_path, 
        pred_class_name, 
        retrieved_class_name, 
        best_sim, 
        pred_conf.item()
    )

if __name__ == '__main__':
    main()