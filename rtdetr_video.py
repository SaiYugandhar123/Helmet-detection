#!/usr/bin/env python3
"""
RT-DETRv2 Video File Inference Script
Process video files with native RT-DETR (no Ultralytics)
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
import sys
import os
import argparse
import time

# Add RT-DETR paths
rtdetr_path = os.path.join(os.path.dirname(__file__), 'RT-DETR', 'rtdetrv2_pytorch')
sys.path.insert(0, rtdetr_path)

# COCO class names (same as webcam script)
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

# Colors for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0)
]

def load_rtdetr_model(config_path, model_path, device='cpu'):
    """Load RT-DETRv2 model"""
    print(f"Loading RT-DETR model...")
    print(f"Device: {device}")
    
    try:
        from src.core import YAMLConfig
        
        cfg = YAMLConfig(config_path, resume=model_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
        
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

def process_video(video_path, model, device, threshold=0.5, output_path=None):
    """Process video file with RT-DETR"""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}FPS, {total_frames} frames")
    
    # Setup video writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Transform for preprocessing
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    
    frame_count = 0
    total_detections = 0
    processing_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            start_time = time.time()
            
            # Preprocess frame
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(rgb_frame).unsqueeze(0).to(device)
            orig_size = torch.tensor([w, h]).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(input_tensor, orig_size)
                labels, boxes, scores = outputs
            
            # Draw detections
            frame_detections = 0
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
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    color = COLORS[label_id % len(COLORS)]
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label_text = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label_text, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    frame_detections += 1
            
            total_detections += frame_detections
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {frame_detections} | Time: {processing_time:.3f}s"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Save frame
            if output_path:
                out.write(frame)
            
            # Show progress
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                avg_time = np.mean(processing_times[-30:])
                print(f"Progress: {progress:.1f}% | Avg time: {avg_time:.3f}s/frame | Detections: {total_detections}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        cap.release()
        if output_path:
            out.release()
        
        # Summary
        avg_processing_time = np.mean(processing_times)
        print(f"\n" + "="*60)
        print("VIDEO PROCESSING SUMMARY")
        print("="*60)
        print(f"Frames processed: {frame_count}/{total_frames}")
        print(f"Total detections: {total_detections}")
        print(f"Average processing time: {avg_processing_time:.3f}s per frame")
        print(f"Average FPS: {1/avg_processing_time:.1f}")
        if output_path:
            print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='RT-DETRv2 Video Processing')
    parser.add_argument('--video', '-v', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', '-m', type=str, default='rtdetrv2_r50vd_m_7x_coco_ema.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--config', '-c', type=str, 
                       default='RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml',
                       help='Path to config file')
    parser.add_argument('--output', '-o', type=str, help='Output video path (optional)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    print("="*60)
    print("RT-DETRv2 Video Processing (Native Framework)")
    print("="*60)
    
    # Load model
    model = load_rtdetr_model(args.config, args.model, args.device)
    if model is None:
        return
    
    # Process video
    output_path = args.output or f"rtdetr_output_{os.path.basename(args.video)}"
    process_video(args.video, model, args.device, args.threshold, output_path)

if __name__ == "__main__":
    main()
