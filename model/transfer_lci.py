import torch
import torch.nn as nn
import timm
import os
import sys

class GatedFeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        g = self.gate(x)
        return x * g + self.proj(x)

class LCI_ConvNeXt(nn.Module):
    def __init__(self, num_classes=102, model_name='convnextv2_base', pretrained_path=None, dropout_rate=0.3):
        super().__init__()
        
        print(f"Initializing Backbone: {model_name}...")
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=False, 
            num_classes=0,
            global_pool='' 
        )
        
        if pretrained_path:
            if os.path.exists(pretrained_path):
                print(f"Loading local weights from: {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                
                if 'model' in checkpoint:
                    checkpoint = checkpoint['model']
                
                current_dict = self.backbone.state_dict()
                new_state_dict = {k: v for k, v in checkpoint.items() if k in current_dict and v.size() == current_dict[k].size()}
                
                self.backbone.load_state_dict(new_state_dict, strict=False)
                print("Pretrained weights loaded successfully.")
            else:
                print(f"Error: Weight file not found at {pretrained_path}")
                sys.exit(1)

        with torch.no_grad():
            dummy = torch.randn(1, 3, 384, 384)
            feat = self.backbone(dummy)
            self.num_features = feat.shape[1]

        self.fusion = GatedFeatureFusion(self.num_features)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.num_features, eps=1e-6),
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fusion(x)
        x = self.global_pool(x)
        x = self.head(x)
        return x

if __name__ == '__main__':
    model = LCI_ConvNeXt(num_classes=102, model_name='convnextv2_base', pretrained_path=None)
    img = torch.randn(2, 3, 384, 384)
    out = model(img)
    print(out.shape)