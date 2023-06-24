import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import numpy as np
from utils import prepare_data_for_training, prepare_data_for_testing, tf_prepare_data_for_training, tf_prepare_data_for_testing
import tensorflow as tf

# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
# If we just use allow_growth_memory in tf then this solution is not working, must be combine with limit percent memory allocation.
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

# Change base dir for our directory
BASE_DIR = r'D:\AI\Computer Vision\projects\face_identifier'

# Setup location of data folder 
TRAIN_DATASET_DIR = os.path.join(BASE_DIR, 'train_dataset')
TEST_DATASET_DIR = os.path.join(BASE_DIR, 'test_dataset')
IMG_FOLDER = 'img'
FACE_FOLDER = 'faces'
EMBEDDING_FOLDER = 'embeddings'

# Check type of device (cpu, gpu, tpu)
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)

# Initalize face detector - MTCNN
face_detector = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device=device)

# Initialize face embedder
# Intialize embedding model
"""face_embedder = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
).to(device)
face_embedder.eval()"""

MODEL_DIR = os.path.join(BASE_DIR, 'model')
model_name = 'face_recognition_triplot.h5'
model_path = os.path.join(MODEL_DIR, model_name)
face_embedder = tf.keras.models.load_model(model_path, compile=False)

if __name__ == '__main__':
    # Extract face from input image
    tf_prepare_data_for_training(face_detector, face_embedder, device)
    tf_prepare_data_for_testing(face_detector, face_embedder, device)