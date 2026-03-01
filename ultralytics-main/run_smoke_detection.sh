#!/bin/bash

# 进入脚本所在目录
cd "$(dirname "$0")"

# 启动吸烟检测应用
cd streamlit_app
python start_app.py 