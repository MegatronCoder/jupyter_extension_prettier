from ultralytics import YOLO

# Load the base YOLOv8n model
model = YOLO('yolov8n.pt')  

# Train the model using your custom data
results = model.train(
    data=r'C:\Users\Manthan\Desktop\CV_DL_Practicals\Persian_Car_Plates_YOLOV8\data.yaml',      # Path to your data.yaml file
    epochs=1,                          # Number of epochs
    imgsz=640,                          # Image size
    batch=16,                           # Batch size
    patience=50,                        # Early stopping patience
    save=True,                          # Save results                         
    workers=8,                          # Number of worker threads
    project='plates_detection',         # Project name
    name='yolov8n_plates'              # Experiment name
)