import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import numpy as np
from utils import prepare_data_for_training, prepare_data_for_testing

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

# Initalize face detector - MTCNN
face_detector = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

# Initialize face embedder
# Intialize embedding model
face_embedder = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
).to(device)
face_embedder.eval()

if __name__ == '__main__':
    # Extract face from input image
    prepare_data_for_training(face_detector, face_embedder, device)
    prepare_data_for_testing(face_detector, face_embedder, device)