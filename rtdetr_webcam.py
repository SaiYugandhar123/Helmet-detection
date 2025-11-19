#!/usr/bin/env python3
"""
Clean RT-DETRv2 Webcam Inference Script
No on-screen text overlays - just pure detection with optional video recording
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

# Color palette for different classes
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 255, 128), (255, 128, 255)
]

class CleanRTDETRWebcam:
    def __init__(self, config_path, model_path, device='auto'):
        # Auto-detect or validate device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"🚀 Auto-detected GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                print("🔧 Auto-detected: CPU (CUDA not available)")
        elif device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = 'cpu'
        elif device == 'cuda' and torch.cuda.is_available():
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            self.device = device
        else:
            self.device = device
            print(f"✓ Using device: {device}")
        
        self.model = self.load_model(config_path, model_path)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def load_model(self, config_path, model_path):
        """Load RT-DETRv2 model"""
        print(f"Loading RT-DETR model...")
        print(f"Device: {self.device}")
        
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
            
            model = RTDETRv2Model().to(self.device)
            model.eval()
            
            # GPU optimizations
            if self.device == 'cuda':
                # Enable mixed precision for better performance
                model.half() if torch.cuda.get_device_capability(0)[0] >= 7 else model
                print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            
            print("✓ RT-DETR model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading RT-DETR model: {e}")
            return None
    
    def preprocess_frame(self, frame):
        """Preprocess webcam frame for RT-DETR"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)
        orig_size = torch.tensor([w, h]).unsqueeze(0).to(self.device)
        return input_tensor, orig_size
    
    def draw_detections(self, frame, labels, boxes, scores, threshold=0.5):
        """Draw detection results on frame - clean version with just bboxes and labels"""
        detection_count = 0
        
        if len(labels) > 0 and len(labels[0]) > 0:
            valid_indices = scores[0] > threshold
            valid_labels = labels[0][valid_indices]
            valid_boxes = boxes[0][valid_indices]
            valid_scores = scores[0][valid_indices]
            
            for i, (label, box, score) in enumerate(zip(valid_labels, valid_boxes, valid_scores)):
                label_id = int(label.item())
                class_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else f"class_{label_id}"
                confidence = score.item()
                
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Choose color based on class
                color = COLORS[label_id % len(COLORS)]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label_text = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 4, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                detection_count += 1
        
        return frame, detection_count
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run_clean_webcam(self, camera_id=0, threshold=0.5, record_video=False, output_path=None):
        """Run clean webcam inference without any on-screen overlays"""
        print("="*70)
        print("Clean RT-DETRv2 Webcam Inference")
        print("="*70)
        print("Clean display mode - no on-screen text overlays")
        print("Stats will be printed to console every 30 frames")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'v' to toggle video recording")
        print("  - Press '+'/'-' to adjust confidence threshold")
        print("="*70)
        
        if self.model is None:
            print("Error: Model not loaded!")
            return
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera initialized: {width}x{height} @ {fps}FPS")
        
        # Video recording setup
        video_writer = None
        is_recording = record_video
        
        if is_recording and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
            print(f"Recording to: {output_path}")
        
        frame_count = 0
        saved_frames = 0
        inference_frequency = 1  # Process every frame
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                frame_count += 1
                detection_count = 0
                
                # Run inference at specified frequency
                if frame_count % inference_frequency == 0:
                    start_time = time.time()
                    
                    input_tensor, orig_size = self.preprocess_frame(frame)
                    
                    with torch.no_grad():
                        # Use autocast for mixed precision on GPU
                        if self.device == 'cuda':
                            with torch.cuda.amp.autocast():
                                outputs = self.model(input_tensor, orig_size)
                        else:
                            outputs = self.model(input_tensor, orig_size)
                        
                        labels, boxes, scores = outputs
                    
                    frame, detection_count = self.draw_detections(frame, labels, boxes, scores, threshold)
                    
                    # Clear GPU cache periodically for memory management
                    if self.device == 'cuda' and frame_count % 100 == 0:
                        torch.cuda.empty_cache()
                    
                    inference_time = time.time() - start_time
                
                # Update FPS
                self.update_fps()
                
                # Console output (every 30 frames)
                if frame_count % 30 == 0:
                    gpu_info = ""
                    if self.device == 'cuda':
                        gpu_memory = torch.cuda.memory_allocated() / 1e6  # MB
                        gpu_info = f" | GPU: {gpu_memory:.0f}MB"
                    
                    print(f"Frame: {frame_count:6d} | FPS: {self.current_fps:4.1f} | "
                          f"Detections: {detection_count:2d} | Threshold: {threshold:.2f} | "
                          f"Recording: {'ON ' if is_recording else 'OFF'}{gpu_info}")
                
                # Minimal recording indicator (tiny red dot in corner)
                if is_recording:
                    cv2.circle(frame, (width - 15, 15), 3, (0, 0, 255), -1)
                
                # Save to video if recording
                if is_recording and video_writer:
                    video_writer.write(frame)
                
                # Display frame
                cv2.imshow('RT-DETRv2 Clean Webcam', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    filename = f'clean_frame_{saved_frames:04d}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame: {filename}")
                    saved_frames += 1
                elif key == ord('v'):
                    if not is_recording and not video_writer:
                        # Start recording
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        new_output_path = f"rtdetr_clean_{timestamp}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(new_output_path, fourcc, 20.0, (width, height))
                        is_recording = True
                        print(f"Started recording: {new_output_path}")
                    else:
                        # Stop recording
                        is_recording = False
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        print("Stopped recording")
                elif key == ord('+') or key == ord('='):
                    threshold = min(0.95, threshold + 0.05)
                    print(f"Threshold: {threshold:.2f}")
                elif key == ord('-'):
                    threshold = max(0.1, threshold - 0.05)
                    print(f"Threshold: {threshold:.2f}")
                elif key == ord('f'):
                    inference_frequency = 2 if inference_frequency == 1 else 1
                    print(f"Inference frequency: every {inference_frequency} frame(s)")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            print("Cleaning up...")
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            print(f"Processed {frame_count} frames total")
            if saved_frames > 0:
                print(f"Saved {saved_frames} frame(s)")

def main():
    parser = argparse.ArgumentParser(description='Clean RT-DETRv2 Webcam Inference')
    parser.add_argument('--model', '-m', type=str, default='rtdetrv2_r50vd_m_7x_coco_ema.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--config', '-c', type=str, 
                       default='RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml',
                       help='Path to config file')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', '-d', type=str, default='auto', help='Device (auto/cpu/cuda)')
    parser.add_argument('--record', action='store_true', help='Start recording immediately')
    parser.add_argument('--output', '-o', type=str, help='Output video path')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Set default output path if recording
    if args.record and not args.output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"rtdetr_clean_{timestamp}.mp4"
    
    # Initialize clean webcam inference
    clean_detector = CleanRTDETRWebcam(args.config, args.model, args.device)
    
    # Run clean webcam inference
    clean_detector.run_clean_webcam(args.camera, args.threshold, args.record, args.output)

if __name__ == "__main__":
    main()
