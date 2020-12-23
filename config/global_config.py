"""
This file contains global configuration settings that are used throughout the project's code
Please change these variables first before considering to adjust specific lines in other files
"""
import os

# Locations
HOME_DIR = os.path.normpath(r"ENTER ABSOLUTE PATH")
DATA_DIR = os.path.normpath(r"/ENTER DATA FOLDER")
MODEL_OUTPUT_DIR = os.path.join(HOME_DIR, 'output', 'output_models')
INPUT_PREDICTION_DIR = os.path.join(HOME_DIR, 'output', 'output_predictions')
OUTPUT_PREDICTION_DIR = os.path.join(HOME_DIR, 'output', 'output_predictions')
PRETRAINED_MODELS_DIR = os.path.normpath(r"PROVIDE PATH TO PRETRAINED MODELS")

# Global variables
SLOT_ANNOTATIONS_FILENAME = "slot_label.txt"
TRAIN_FOLDER_NAME = "train"
VAL_FOLDER_NAME = "val"
TEST_FOLDER_NAME = "test"
CUDA_PREF = 0

# RNN specific variables
RNN_VOCAB_DOMAIN = "atis"
