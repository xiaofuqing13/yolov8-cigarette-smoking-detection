#!/bin/bash

# 进入脚本所在目录
cd "$(dirname "$0")"

# 启动Streamlit应用使用自定义启动脚本
echo "启动吸烟检测系统..."
echo "请在浏览器中访问: http://localhost:8501"
python start_app.py 