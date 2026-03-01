# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from ultralytics import RTDETR, YOLO
from ultralytics.utils import ASSETS, DEFAULT_CFG, LINUX, MACOS, ONLINE, ROOT, SETTINGS, WINDOWS

CFG = r'E:\yolov8\ultralytics-main\ultralytics\cfg\models\v8\CA.yaml'
SOURCE = ASSETS / 'bus.jpg'


def test_model_forward():
    model = YOLO(CFG)
    model(SOURCE)  # also test no source and augment