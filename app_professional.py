"""
Professional Mental Stress Detection App
Modern UI with interactive elements and professional design
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import json
from typing import Dict, List, Optional
import sys
from datetime import datetime
import uuid
import time

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import preprocess_text
from feature_engineering import FeatureExtractor
