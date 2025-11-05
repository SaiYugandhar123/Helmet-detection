from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

results = model.train(
    data="C:/Users/z00543ct/Desktop/helmet/data.yaml", 
    epochs=50,              # Good balance for CPU
    imgsz=640,             
    batch=16,              
    name="predict",        
    device='cpu',
    freeze=5,              # 🚀 FREEZE: First 5 layers (not 10)
    lr0=0.001,              # 🚀 ADDED: Lower learning rate for frozen model
    patience=15,            # 🚀 ADDED: Early stopping
    save_period=5,         # 🚀 ADDED: Save every 5 epochs (not 10)
    cache=True,             # 🚀 ADDED: Cache for faster training
    workers=4               # 🚀 ADDED: Use multiple CPU cores
)