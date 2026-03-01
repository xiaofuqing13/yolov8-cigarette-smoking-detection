# 吸烟检测系统

这是一个基于YOLOv8和Streamlit开发的吸烟检测系统，可以对图片、视频和摄像头实时画面进行吸烟行为检测。

## 功能特点

- 用户登录界面（默认用户名：admin，密码：password）
- 图片吸烟检测
- 视频吸烟检测
- 摄像头实时吸烟检测
- 可调节的检测置信度阈值

## 安装要求

确保您已安装以下依赖：

```bash
pip install streamlit opencv-python pillow numpy ultralytics
```

## 如何使用

1. 在终端中执行以下命令启动应用程序：

```bash
cd streamlit_app
chmod +x run.sh  # 赋予执行权限
./run.sh
```

或者直接运行：

```bash
cd streamlit_app
streamlit run app.py
```

2. 在浏览器中访问 http://localhost:8501

3. 使用默认凭据登录：
   - 用户名：admin
   - 密码：password

4. 选择所需的检测模式（图片/视频/摄像头）并按照界面提示操作。

## 系统截图

### 登录界面
![登录界面](https://via.placeholder.com/800x450.png?text=登录界面)

### 图片检测
![图片检测](https://via.placeholder.com/800x450.png?text=图片检测)

### 视频检测
![视频检测](https://via.placeholder.com/800x450.png?text=视频检测)

### 摄像头检测
![摄像头检测](https://via.placeholder.com/800x450.png?text=摄像头检测)

## 注意事项

- 摄像头检测需要浏览器权限，请确保已授予访问摄像头的权限
- 视频处理可能需要一些时间，取决于视频长度和复杂度
- 模型使用的是已训练好的YOLOv8模型（runs/detect/train28/weights/best.pt） 