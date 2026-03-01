import streamlit as st

# 设置页面标题和布局 - 必须是第一个Streamlit命令
st.set_page_config(
    page_title="吸烟检测系统",
    page_icon="🚭",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import cv2
import numpy as np

# 在导入ultralytics之前禁用信号处理
import signal
# 保存原始signal.signal函数
original_signal = signal.signal
# 创建一个替代函数，忽略所有信号注册尝试
def dummy_signal(signalnum, handler):
    pass
# 替换signal.signal函数
signal.signal = dummy_signal

# 导入并应用ECA模块补丁，使用print而不是st.info来避免在set_page_config前调用Streamlit命令
try:
    from custom_modules import patch_ultralytics
    patch_success = patch_ultralytics()
    # 补丁应用状态将在加载模型时检查
except Exception as e:
    print(f"加载自定义模块时出错: {str(e)}")

from ultralytics import YOLO
import tempfile
from PIL import Image
import pandas as pd
import time

# 恢复原始signal.signal函数(如果需要)
# signal.signal = original_signal

# 加载CSS样式
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 设置默认用户名和密码
DEFAULT_USERNAME = "1"
DEFAULT_PASSWORD = "1"

# 加载YOLOv8模型
@st.cache_resource
def load_model():
    # 定义模型路径
    custom_model_path = '/Users/xiaofuqing/Downloads/yolov8/ultralytics-main/runs/detect/train28/weights/best.pt'
    
    # 显示ECA补丁应用状态
    if 'patch_success' in globals() and patch_success:
        st.success("成功应用ECA模块补丁")
    
    try:
        # 尝试加载吸烟检测模型
        st.info(f"尝试加载吸烟检测模型: {custom_model_path}")
        
        # 检查文件是否存在
        if not os.path.exists(custom_model_path):
            st.error(f"错误: 模型文件不存在: {custom_model_path}")
            raise FileNotFoundError(f"模型文件不存在: {custom_model_path}")
        
        # 尝试使用常规方式加载模型
        try:
            model = YOLO(custom_model_path)
            st.success("成功加载吸烟检测模型！")
            return model
        except Exception as normal_load_error:
            st.warning(f"常规方式加载模型失败: {str(normal_load_error)}")
            
            # 尝试使用torch直接加载模型（跳过ultralytics封装）
            try:
                import torch
                from pathlib import Path
                
                st.info("尝试使用兼容模式加载模型...")
                
                # 直接使用YOLOv8n作为基础模型并加载自定义权重
                model = YOLO('yolov8n.pt')
                
                # 使用自定义方式加载权重
                # 这是模拟加载过程，实际效果取决于模型结构
                try:
                    weights = torch.load(custom_model_path, map_location='cpu')
                    # 尝试使用safe_load替代方式
                    model.model.load(weights)
                    st.success("成功使用兼容模式加载吸烟检测模型！")
                    return model
                except Exception as weight_error:
                    raise Exception(f"加载模型权重失败: {str(weight_error)}")
            except Exception as compat_error:
                raise Exception(f"兼容方式加载失败: {str(compat_error)}")
        
    except Exception as e:
        st.error(f"加载模型失败: {str(e)}")
        
        # 根据用户要求，不加载通用模型
        st.error("无法加载吸烟检测模型")
        st.markdown("""
        ### 推荐解决方案:
        1. 请确认模型是使用哪个版本的ultralytics库训练的
        2. 尝试安装相同版本: `pip install ultralytics==版本号`
        3. 或者重新训练模型，使用当前环境的ultralytics版本
        4. 如有可能，可以联系模型的训练者获取兼容信息
        """)
        return None

def login_page():
    st.title("欢迎使用吸烟检测系统")
    st.markdown("### 请登录")
    
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    
    login_button = st.button("登录")
    
    if login_button:
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            st.session_state.logged_in = True
            st.success("登录成功！")
            st.rerun()
        else:
            st.error("用户名或密码错误，请重试！")

def main_app():
    model = load_model()
    
    # 如果模型加载失败，显示错误消息并提供解决方案
    if model is None:
        st.error("无法加载模型")
        st.markdown("""
        ### 解决方案:
        1. 确认模型文件路径是否正确
        2. 确认模型文件是否存在
        3. 请联系管理员获取帮助
        """)
        
        # 退出登录按钮
        if st.button("退出登录"):
            st.session_state.logged_in = False
            st.rerun()
        return
    
    st.sidebar.title("吸烟检测系统")
    st.sidebar.markdown("使用YOLOv8进行实时吸烟行为检测")
    
    # 显示模型信息
    st.sidebar.markdown("---")
    with st.sidebar.expander("模型信息", expanded=False):
        # 安全获取模型信息
        try:
            model_type = type(model).__name__
            st.write(f"模型类型: {model_type}")
            
            # 尝试获取模型类别
            if hasattr(model, 'names') and model.names:
                class_names = ', '.join(model.names.values())
                st.write(f"检测类别: {class_names}")
            else:
                st.write("检测类别: 未知")
                
            # 显示模型任务类型
            if hasattr(model, 'task'):
                st.write(f"任务类型: {model.task}")
            
            # 显示模型版本信息
            if hasattr(model, 'info') and model.info:
                st.write(f"版本信息: {model.info}")
        except Exception as e:
            st.write("无法获取详细模型信息")
            st.write(f"模型类型: {type(model).__name__}")
    
    # 创建页面选择器
    app_mode = st.sidebar.selectbox(
        "选择检测模式",
        ["图片检测", "视频检测", "摄像头检测"]
    )
    
    st.sidebar.markdown("---")
    confidence = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("退出登录"):
        st.session_state.logged_in = False
        st.rerun()
    
    # 根据选择的模式显示相应的页面
    if app_mode == "图片检测":
        image_detection_page(model, confidence)
    elif app_mode == "视频检测":
        video_detection_page(model, confidence)
    elif app_mode == "摄像头检测":
        camera_detection_page(model, confidence)

def image_detection_page(model, confidence):
    st.title("图片吸烟检测")
    uploaded_file = st.file_uploader("上传图片...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 读取上传的图片并转换为PIL图像
        image = Image.open(uploaded_file)
        
        # 将PIL图像转换为OpenCV格式
        image_np = np.array(image)
        
        if st.button("开始检测"):
            # 使用YOLOv8进行预测
            results = model(image_np, conf=confidence)
            
            # 可视化结果
            result_image = results[0].plot()
            
            # 显示结果
            st.image(result_image, caption="检测结果", use_column_width=True)
            
            # 显示检测到的对象信息
            if len(results[0].boxes) > 0:
                st.markdown("### 检测结果")
                
                # 创建表格显示检测结果
                data = []
                for box in results[0].boxes:
                    conf = box.conf.item()
                    cls = box.cls.item()
                    cls_name = results[0].names[int(cls)]
                    data.append({"类别": cls_name, "置信度": f"{conf:.2f}"})
                
                # 显示检测到的物体数量
                st.write(f"检测到 {len(results[0].boxes)} 个物体")
                
                # 如果我们使用的是通用模型，显示所有检测结果
                # 创建数据框并显示
                df = pd.DataFrame(data)
                st.dataframe(df)
                
                # 统计各类别数量
                class_counts = {}
                for item in data:
                    class_name = item["类别"]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                
                # 显示类别统计
                st.markdown("### 类别统计")
                for cls_name, count in class_counts.items():
                    st.write(f"{cls_name}: {count} 个")
            else:
                st.write("未检测到任何物体")
    else:
        # 显示示例信息
        st.info("请上传图片以检测吸烟行为")
        st.markdown("""
        ### 功能说明
        上传图片后，系统将自动检测图中是否有人在吸烟，并标记出吸烟者的位置。
        调整侧边栏中的置信度阈值可以控制检测的灵敏度。
        """)

def video_detection_page(model, confidence):
    st.title("视频吸烟检测")
    uploaded_file = st.file_uploader("上传视频...", type=["mp4", "avi", "mov"])
    
    # 显示支持格式提示
    st.caption("支持的视频格式: MP4, AVI, MOV")
    
    if uploaded_file is not None:
        try:
            # 保存上传的视频到临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name
            
            # 检查视频是否可以打开
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("无法打开视频文件，请确保视频格式正确。")
                if os.path.exists(video_path):
                    os.unlink(video_path)
                return
            cap.release()
            
            # 显示上传的视频
            st.video(video_path)
            
            if st.button("开始检测"):
                with st.spinner("视频处理中，请耐心等待..."):
                    # 处理视频文件
                    cap = cv2.VideoCapture(video_path)
                    
                    # 获取视频的帧率、宽度和高度
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # 创建临时输出视频文件
                    output_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    # 创建进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 获取视频总帧数
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    processed_frames = 0
                    
                    # 处理视频的每一帧
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # 使用YOLOv8进行预测
                        results = model(frame, conf=confidence)
                        
                        # 绘制结果
                        result_frame = results[0].plot()
                        
                        # 写入输出视频
                        out.write(result_frame)
                        
                        # 更新进度条
                        processed_frames += 1
                        progress = processed_frames / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"处理进度: {processed_frames}/{total_frames} 帧 ({progress:.1%})")
                    
                    # 释放资源
                    cap.release()
                    out.release()
                    
                    # 显示处理后的视频
                    st.success("视频处理完成！")
                    st.video(output_path)
            
            # 清理临时文件
            if os.path.exists(video_path):
                os.unlink(video_path)
                
        except Exception as e:
            st.error(f"处理视频时出错: {str(e)}")
            st.warning("请尝试上传不同的视频文件或降低视频分辨率。")
    else:
        # 显示示例信息
        st.info("请上传视频文件以检测吸烟行为")
        st.markdown("""
        ### 功能说明
        上传视频后，系统将分析视频中的每一帧，识别出吸烟行为并进行标记。
        处理完成后会生成带有检测标记的新视频。
        """)

def camera_detection_page(model, confidence):
    st.title("摄像头吸烟检测")
    
    # 设置STUN服务器以帮助穿透NAT
    st.markdown("""
    ### 注意
    摄像头检测需要允许浏览器访问您的摄像头。请确保您已经授予相应的权限。
    """)
    
    # 开始/停止摄像头按钮
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("开始检测", key="start_cam")
    with col2:
        stop_button = st.button("停止检测", key="stop_cam")
    
    # 创建占位符用于显示摄像头画面
    video_placeholder = st.empty()
    
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    if start_button:
        st.session_state.camera_running = True
    
    if stop_button:
        st.session_state.camera_running = False
    
    if st.session_state.camera_running:
        try:
            cap = cv2.VideoCapture(0)  # 使用默认摄像头
            
            # 检查摄像头是否成功打开
            if not cap.isOpened():
                st.error("无法访问摄像头，请检查摄像头连接或权限设置。")
                st.session_state.camera_running = False
                st.markdown("""
                ### 可能的原因:
                1. 摄像头已被其他应用程序占用
                2. 浏览器没有获得摄像头访问权限
                3. 摄像头驱动程序问题
                
                请关闭可能使用摄像头的其他应用，然后重试。
                """)
            else:
                st.info("摄像头已启动，正在进行实时检测...")
                
                frame_placeholder = st.empty()
                
                # 显示FPS计数器
                fps_text = st.empty()
                start_time = time.time()
                frame_count = 0
                
                # 使用try-finally确保摄像头资源被释放
                try:
                    while st.session_state.camera_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("无法从摄像头获取画面")
                            break
                        
                        # 使用安全的预测方式
                        try:
                            # 使用YOLOv8进行预测
                            results = model(frame, conf=confidence)
                            
                            # 绘制结果
                            result_frame = results[0].plot()
                            
                            # RGB转换（OpenCV使用BGR，而Streamlit需要RGB）
                            result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                        except Exception as predict_error:
                            st.error(f"预测过程出错: {str(predict_error)}")
                            # 如果预测失败，显示原始画面
                            result_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # 显示图像
                        frame_placeholder.image(result_frame_rgb, caption="实时检测", use_column_width=True)
                        
                        # 计算并显示FPS
                        frame_count += 1
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 1.0:  # 每秒更新一次FPS
                            fps = frame_count / elapsed_time
                            fps_text.text(f"FPS: {fps:.2f}")
                            frame_count = 0
                            start_time = time.time()
                finally:
                    # 释放摄像头
                    cap.release()
                    st.info("摄像头已停止")
        except Exception as e:
            st.error(f"摄像头检测发生错误: {str(e)}")
            # 显示详细的错误信息和堆栈跟踪
            import traceback
            st.code(traceback.format_exc(), language="python")
            st.session_state.camera_running = False
            
            # 提供解决方案
            st.warning("""
            ### 可能的解决方案:
            1. 刷新页面后重试
            2. 尝试使用图片检测模式代替
            3. 确保您的摄像头正常工作
            """)
    else:
        st.info("点击'开始检测'按钮启动摄像头进行实时吸烟行为检测")
        st.markdown("""
        ### 功能说明
        启动摄像头后，系统将实时检测画面中的吸烟行为。
        这适用于实时监控场景，如公共区域的吸烟检测。
        """)

# 主函数
def main():
    # 加载CSS样式
    try:
        load_css()
    except Exception as e:
        st.warning("无法加载自定义样式，将使用默认样式。")
    
    # 初始化session状态
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # 添加帮助信息
    with st.sidebar:
        if st.button("帮助"):
            st.info("""
            ### 使用说明:
            1. 首先使用用户名和密码登录系统
            2. 选择需要的检测模式（图片/视频/摄像头）
            3. 调整置信度阈值以获得更准确的吸烟行为检测结果
            4. 上传媒体文件或启动摄像头以进行检测
            
            此系统专门用于检测吸烟行为，可用于公共场所吸烟监控。
            
            如有任何问题，请联系管理员。
            """)
    
    # 判断是否已登录
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()

if __name__ == "__main__":
    main() 