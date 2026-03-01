from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")
model = YOLO(r"E:\yolov8\ultralytics-main\ultralytics\cfg\models\v8\CA.yaml")

# Train the model
train_results = model.train(
    data=r"E:\yolov8\ultralytics-main\ultralytics\cfg\datasets\smoke.yaml",  # path to dataset YAML
    #model=r"E:\yolov8\ultralytics-main\ultralytics\cfg\models\v8\shufflenetv2.yaml",
    workers=0,
    epochs=100,
    batch=8,  # number of training epochs
    device="",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)
