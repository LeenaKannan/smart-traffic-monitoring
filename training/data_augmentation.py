# training/data_augmentation.py
import albumentations as A
import cv2
import numpy as np
from pathlib import Path
import json

class IndianTrafficAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            # Weather conditions common in India
            A.RandomRain(p=0.3),
            A.RandomFog(p=0.2),
            A.RandomSunFlare(p=0.2),
            
            # Lighting variations
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            
            # Motion blur for moving vehicles
            A.MotionBlur(p=0.3),
            
            # Noise (common in traffic cameras)
            A.GaussNoise(p=0.2),
            
            # Geometric transformations
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            
            # Occlusion simulation
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def augment_dataset(self, image_dir: str, label_dir: str, output_dir: str, multiplier: int = 3):
        """Augment Indian traffic dataset"""
        image_path = Path(image_dir)
        label_path = Path(label_dir)
        output_path = Path(output_dir)
        
        output_path.mkdir(exist_ok=True)
        (output_path / 'images').mkdir(exist_ok=True)
        (output_path / 'labels').mkdir(exist_ok=True)
        
        for img_file in image_path.glob('*.jpg'):
            # Load image
            image = cv2.imread(str(img_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load corresponding label
            label_file = label_path / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
                
            bboxes, class_labels = self._load_yolo_labels(label_file)
            
            # Generate augmented versions
            for i in range(multiplier):
                try:
                    transformed = self.transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    # Save augmented image
                    aug_img_name = f"{img_file.stem}_aug_{i}.jpg"
                    aug_img_path = output_path / 'images' / aug_img_name
                    
                    aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_img_path), aug_image)
                    
                    # Save augmented labels
                    aug_label_name = f"{img_file.stem}_aug_{i}.txt"
                    aug_label_path = output_path / 'labels' / aug_label_name
                    
                    self._save_yolo_labels(
                        aug_label_path,
                        transformed['bboxes'],
                        transformed['class_labels']
                    )
                    
                except Exception as e:
                    print(f"Error augmenting {img_file}: {e}")
                    continue

    def _load_yolo_labels(self, label_file):
        """Load YOLO format labels"""
        bboxes = []
        class_labels = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
        
        return bboxes, class_labels

    def _save_yolo_labels(self, label_file, bboxes, class_labels):
        """Save YOLO format labels"""
        with open(label_file, 'w') as f:
            for bbox, class_id in zip(bboxes, class_labels):
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
