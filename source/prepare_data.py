import cv2
from facenet_pytorch import MTCNN
import torch
import os
import numpy as np

# Change base dir for our directory
BASE_DIR = '/content/drive/MyDrive/projects/ComputerVision/identifier_system_by_face'

# Setup location of data folder 
TRAIN_DATASET_DIR = os.path.join(BASE_DIR, 'train_dataset')
TEST_DATASET_DIR = os.path.join(BASE_DIR, 'test_dataset')
IMG_FOLDER = 'img'
FACE_FOLDER = 'faces'
EMBEDDING_FOLDER = 'embeddings'

# Check type of device (cpu, gpu, tpu)
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

# Extract face from input image
