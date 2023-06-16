import cv2
from facenet_pytorch import MTCNN
import torch
import os

#check type of device (cpu, gpu, tpu)
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)