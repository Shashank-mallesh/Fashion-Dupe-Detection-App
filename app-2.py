"""
Fashion Dupe Detection - Streamlit Application
"""
import os
import streamlit as st

# Install system dependencies for OpenCV
try:
    import cv2
except ImportError:
    st.error("OpenCV not available. This app requires OpenCV for image processing.")

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Rest of your code remains the same...
from model import load_model
from utils import predict_class, compute_similarity, is_dupe

# Page configuration
st.set_page_config(
    page_title="Fashion Dupe Detection",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS and the rest of your code...
