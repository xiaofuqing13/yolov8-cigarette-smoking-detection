from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("/Users/xiaofuqing/Downloads/yolov8/ultralytics-main/runs/detect/train28/weights/best.pt")

# Define path to the image file
source = "/Users/xiaofuqing/Downloads/yolov8/ultralytics-main/dataset/smoke/images/train/0001.jpg"

# Run inference on the source
results = model(source,save=True)  # list of Results objects