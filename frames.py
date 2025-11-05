import cv2
import os

def video_to_frames(video_path, output_folder, fps=5):
    """
    Convert video to frames
    
    Args:
        video_path: Path to input video file
        output_folder: Folder to save extracted frames
        fps: Frames per second to extract (5 = 5 frames per second)
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    print(f"Video FPS: {video_fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting {fps} frame(s) per second...")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame at specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"hiz_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            print(f"Saved: {frame_filename}")
        
        frame_count += 1
    
    cap.release()
    print(f"\nExtraction complete!")
    print(f"Total frames extracted: {saved_count}")
    print(f"Frames saved in: {output_folder}")

# Configuration
video_path = "demovid4.mp4"              # Input video file
output_folder = "extracted_frames"       # Output folder for frames
frames_per_second = 5                   # Extract 5 frames per second

# Run the conversion
video_to_frames(video_path, output_folder, frames_per_second)