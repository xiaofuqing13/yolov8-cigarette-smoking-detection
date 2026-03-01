"""
吸烟检测系统启动脚本
在导入ultralytics前设置环境变量并启动Streamlit应用
"""

import os
import sys

# 设置环境变量以禁用ultralytics的hub功能和信号处理
os.environ["ULTRALYTICS_HUB"] = "0"  # 禁用hub功能
os.environ["YOLO_VERBOSE"] = "0"     # 减少输出信息
os.environ["DISABLE_SIGNAL_HANDLERS"] = "1"  # 禁用信号处理

# 导入streamlit并启动应用
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")
    
    print("启动吸烟检测系统...")
    print(f"应用路径: {app_path}")
    print("设置环境变量已完成，正在启动应用...")
    
    # 使用8502端口（或其他未被占用的端口）
    sys.argv = ["streamlit", "run", app_path, "--server.port=8502"]
    sys.exit(stcli.main()) 