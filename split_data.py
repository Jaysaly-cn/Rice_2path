import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_root, target_root, split_ratio=(0.8, 0.1, 0.1), seed=2024):

    random.seed(seed)

    if sum(split_ratio) != 1.0:
        print("Error: Split ratios must sum to 1.0")
        return

    classes = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    classes.sort()
    
    print(f"Found {len(classes)} classes: {classes}")

    for split in ['train', 'valid', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(target_root, split, cls), exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(source_root, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(images)
        
        total_img = len(images)
        train_end = int(total_img * split_ratio[0])
        valid_end = train_end + int(total_img * split_ratio[1])
        
        splits = {
            'train': images[:train_end],
            'valid': images[train_end:valid_end],
            'test':  images[valid_end:]
        }
        
        print(f"Processing {cls}: Total {total_img} -> Train {len(splits['train'])}, Valid {len(splits['valid'])}, Test {len(splits['test'])}")
        
        for split_name, img_list in splits.items():
            for img_name in tqdm(img_list, desc=f"Copying {cls} to {split_name}", leave=False):
                src_path = os.path.join(cls_dir, img_name)
                dst_path = os.path.join(target_root, split_name, cls, img_name)
                shutil.copy2(src_path, dst_path)

    print("\n? Dataset split completed successfully!")
    print(f"Data saved to: {target_root}")

if __name__ == '__main__':

    original_dataset_path = "./data/soybean176_raw"
    
    target_dataset_path = "./data/soybean176"
    
    ratios = (0.8, 0.1, 0.1)
    
    
    if not os.path.exists(original_dataset_path):
        print(f"Error: Source directory '{original_dataset_path}' does not exist.")
        print("Please create it and put your class folders inside.")
    else:
        split_dataset(original_dataset_path, target_dataset_path, ratios)