#!/usr/bin/env python3
"""
RT-DETRv2 Batch Image Processing Script
Process multiple images with native RT-DETR (no Ultralytics)
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import sys
import os
import argparse
import glob
import time

# Add RT-DETR paths
rtdetr_path = os.path.join(os.path.dirname(__file__), 'RT-DETR', 'rtdetrv2_pytorch')
sys.path.insert(0, rtdetr_path)

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def load_rtdetr_model(config_path, model_path, device='cpu'):
    """Load RT-DETRv2 model"""
    print(f"Loading RT-DETR model...")
    
    try:
        from src.core import YAMLConfig
        
        cfg = YAMLConfig(config_path, resume=model_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
            print("✓ Using EMA weights")
        else:
            state = checkpoint['model']
            print("✓ Using model weights")
        
        cfg.model.load_state_dict(state)
        
        class RTDETRv2Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
                
            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs
        
        model = RTDETRv2Model().to(device)
        model.eval()
        
        print("✓ RT-DETR model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def process_single_image(image_path, model, device, threshold=0.5, output_dir='output'):
    """Process a single image"""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    im_data = transforms(image)[None].to(device)
    orig_size = torch.tensor([w, h])[None].to(device)
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(im_data, orig_size)
        labels, boxes, scores = outputs
    inference_time = time.time() - start_time
    
    # Process results
    detections = []
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    if len(labels) > 0 and len(labels[0]) > 0:
        valid_indices = scores[0] > threshold
        valid_labels = labels[0][valid_indices]
        valid_boxes = boxes[0][valid_indices]
        valid_scores = scores[0][valid_indices]
        
        for label, box, score in zip(valid_labels, valid_boxes, valid_scores):
            label_id = int(label.item())
            class_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else f"class_{label_id}"
            confidence = score.item()
            
            x1, y1, x2, y2 = box.cpu().numpy()
            
            # Store detection info
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            
            # Draw label
            label_text = f"{class_name}: {confidence:.2f}"
            draw.text((x1, y1 - 20), label_text, fill='red', font=font)
    
    # Save result
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
    result_image.save(output_path)
    
    return detections, inference_time, output_path

def process_batch(image_paths, model, device, threshold=0.5, output_dir='batch_output'):
    """Process batch of images"""
    
    print(f"Processing {len(image_paths)} images...")
    print("="*60)
    
    total_detections = 0
    total_time = 0
    results = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image_path in enumerate(image_paths):
        try:
            print(f"Processing [{i+1}/{len(image_paths)}]: {os.path.basename(image_path)}")
            
            detections, inference_time, output_path = process_single_image(
                image_path, model, device, threshold, output_dir
            )
            
            total_detections += len(detections)
            total_time += inference_time
            
            result = {
                'image': image_path,
                'output': output_path,
                'detections': len(detections),
                'inference_time': inference_time,
                'objects': detections
            }
            results.append(result)
            
            print(f"  ✓ Found {len(detections)} objects in {inference_time:.3f}s")
            
            # Print top detections
            if detections:
                top_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:3]
                for det in top_detections:
                    print(f"    - {det['class']}: {det['confidence']:.3f}")
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path}: {e}")
            continue
    
    # Summary
    avg_time = total_time / len(image_paths) if image_paths else 0
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Images processed: {len(results)}/{len(image_paths)}")
    print(f"Total detections: {total_detections}")
    print(f"Average inference time: {avg_time:.3f}s per image")
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Output directory: {output_dir}")
    
    # Save results summary
    summary_path = os.path.join(output_dir, 'detection_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RT-DETRv2 Batch Processing Results\n")
        f.write("="*50 + "\n\n")
        
        for result in results:
            f.write(f"Image: {result['image']}\n")
            f.write(f"Detections: {result['detections']}\n")
            f.write(f"Inference time: {result['inference_time']:.3f}s\n")
            f.write("Objects:\n")
            
            for obj in result['objects']:
                f.write(f"  - {obj['class']}: {obj['confidence']:.3f}\n")
            f.write("\n")
    
    print(f"✓ Results summary saved to: {summary_path}")
    
    return results

def find_images(input_path):
    """Find image files in directory or return single file"""
    
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    if os.path.isfile(input_path):
        return [input_path]
    
    elif os.path.isdir(input_path):
        image_files = []
        for ext in supported_formats:
            pattern = os.path.join(input_path, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(input_path, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
        
        return sorted(list(set(image_files)))  # Remove duplicates and sort
    
    else:
        # Try glob pattern
        return sorted(glob.glob(input_path))

def main():
    parser = argparse.ArgumentParser(description='RT-DETRv2 Batch Image Processing')
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='Input image file, directory, or glob pattern')
    parser.add_argument('--model', '-m', type=str, default='rtdetrv2_r50vd_m_7x_coco_ema.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--config', '-c', type=str, 
                       default='RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml',
                       help='Path to config file')
    parser.add_argument('--output', '-o', type=str, default='batch_output', 
                       help='Output directory')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, 
                       help='Confidence threshold')
    parser.add_argument('--device', '-d', type=str, default='cpu', 
                       help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Find input images
    image_files = find_images(args.input)
    
    if not image_files:
        print(f"Error: No image files found in {args.input}")
        return
    
    print(f"Found {len(image_files)} image(s)")
    
    # Check model files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    print("="*60)
    print("RT-DETRv2 Batch Image Processing (Native Framework)")
    print("="*60)
    
    # Load model
    model = load_rtdetr_model(args.config, args.model, args.device)
    if model is None:
        return
    
    # Process images
    results = process_batch(image_files, model, args.device, args.threshold, args.output)
    
    print("\n✓ Batch processing completed!")

if __name__ == "__main__":
    main()
