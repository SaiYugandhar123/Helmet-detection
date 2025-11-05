import os
from ultralytics import YOLO

model_path = "best.pt"  
input_folder = "demovid0.mp4"   
output_folder = "C:/Users/z00543ct/Desktop/helmet/output4" 
os.makedirs(output_folder, exist_ok=True)
model = YOLO(model_path)
results = model.predict(
    source=input_folder,
    save=True,              
    save_txt=False,          
    project=output_folder, 
    name="",                
    exist_ok=True,
)

print(f"Inference complete! Check results in: {output_folder}")
