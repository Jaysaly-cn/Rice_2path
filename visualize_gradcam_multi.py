import os
import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from tqdm import tqdm
import matplotlib

from model.transfer_lci import LCI_ConvNeXt
from utils.datasets import RicePestDataset

matplotlib.use('Agg')

def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        self.handlers.append(self.target_layer.register_forward_hook(self.save_activation))
        self.handlers.append(self.target_layer.register_full_backward_hook(self.save_gradient))

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
        self.handlers = []

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        conf = torch.softmax(output, dim=1)[0, class_idx].item()
        return cam.data.cpu().numpy()[0, 0], conf

def get_structure_image(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    return cv2.cvtColor(l_channel, cv2.COLOR_GRAY2RGB)

def get_color_block_image(img_rgb):
    return cv2.GaussianBlur(img_rgb, (51, 51), 0)

def apply_heatmap(bg_img, cam_map):
    heatmap = cv2.resize(cam_map, (bg_img.shape[1], bg_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(bg_img, 0.6, heatmap_color, 0.4, 0)

def generate_advanced_grid(model, dataset, device, output_path='gradcam_2path_vis.png'):
    class_names = dataset.classes
    num_classes = len(class_names)
    
    target_layers = [
        ("Global Attention", model.backbone.stages[-1], 'original'),
        ("Path 1: Structure (LAB)", model.fusion.gate[0], 'structure'), 
        ("Path 2: Color (RGB)", model.fusion.proj, 'color')
    ]
    
    fig, axes = plt.subplots(num_classes, 4, figsize=(20, 5 * num_classes))
    if num_classes == 1: axes = np.expand_dims(axes, axis=0)

    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
        
    print(f"Generating advanced heatmaps for {num_classes} classes...")

    for cls_idx in tqdm(range(num_classes)):
        indices = class_indices[cls_idx]
        if not indices:
            continue
            
        rand_idx = random.choice(indices)
        img_path, true_label_idx = dataset.samples[rand_idx]
        true_label = class_names[true_label_idx]
        
        img_raw = cv2.imread(img_path)
        if img_raw is None: continue
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_input = dataset.transform_logic(img_path).unsqueeze(0).to(device)
        
        axes[cls_idx, 0].imshow(img_rgb)
        axes[cls_idx, 0].set_title(f"Original Input\n{true_label}", fontweight='bold', fontsize=14)
        axes[cls_idx, 0].axis('off')
        
        with torch.no_grad():
            output = model(img_input)
            probs = torch.softmax(output, dim=1)
            pred_conf, pred_idx = torch.max(probs, 1)
            pred_label = class_names[pred_idx.item()]
            pred_conf_val = pred_conf.item()

        for i, (title, target_layer, bg_type) in enumerate(target_layers):
            grad_cam = GradCAM(model, target_layer)
            cam_map, _ = grad_cam(img_input, class_idx=pred_idx.item())
            grad_cam.remove_hooks()
            
            if bg_type == 'original':
                bg_img = img_rgb
            elif bg_type == 'structure':
                bg_img = get_structure_image(img_rgb)
            elif bg_type == 'color':
                bg_img = get_color_block_image(img_rgb)
            
            overlay = apply_heatmap(bg_img, cam_map)
            
            col_idx = i + 1
            axes[cls_idx, col_idx].imshow(overlay)
            
            title_color = 'darkgreen' if pred_idx.item() == true_label_idx else 'darkred'
            full_title = f"{title}\nConf: {pred_conf_val:.1%}"
            axes[cls_idx, col_idx].set_title(full_title, color=title_color, fontsize=12, fontweight='bold')
            axes[cls_idx, col_idx].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Advanced visualization saved to {output_path}")

def main():
    config = load_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    test_dir = os.path.join(config['data_root'], 'test')
    test_ds = RicePestDataset(test_dir, is_training=False, image_size=config['input_size'])
    config['num_classes'] = len(test_ds.classes)
    
    model = LCI_ConvNeXt(
        num_classes=config['num_classes'], 
        model_name=config['backbone'],
        pretrained_path=None,
        dropout_rate=config['dropout_rate']
    )
    
    weight_path = os.path.join(config['checkpoints_dir'], 'best_model_ema.pth')
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        print(f"Loaded EMA weights.")
    else:
        weight_path_reg = os.path.join(config['checkpoints_dir'], 'best_model.pth')
        if os.path.exists(weight_path_reg):
             model.load_state_dict(torch.load(weight_path_reg, map_location='cpu'))
        else:
            print("Warning: No weights found. Visualizing untrained model (random noise).")

    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = True
    
    generate_advanced_grid(model, test_ds, device)

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    main()