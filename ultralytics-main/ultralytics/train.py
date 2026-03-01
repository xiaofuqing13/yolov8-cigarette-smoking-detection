from ultralytics import YOLO

# 加载模型
# model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练模型（推荐用于训练）
if __name__ == '__main__':
# Use the model
    results = model.train(data="E:\yolov8\\ultralytics-main\\ultralytics\cfg\datasets\person.yaml", epochs=50, batch=4,device=cpu)             # 训练模型
