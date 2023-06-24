import cv2
from utils import extract_face,load_embedding_dataset_for_deploy, inference, tf_load_embedding_dataset_for_deploy, tf_inference
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchsummary import summary
import time
import os
import tensorflow as tf

# Check type of device (cpu, gpu, tpu)
device =  torch.device('cuda:0' if torch.cuda.is_available() == True else 'cpu')
print('Using device: ', device)

# Initalize face detector - MTCNN
mtcnn = MTCNN(margin = 20, keep_all=False, post_process=False, device=device)

#Intialize embedding model
# model = InceptionResnetV1(
#     classify=False,
#     pretrained="vggface2"
# ).to(device)
# model.eval()
# summary(model, (3, 160, 160))

# Limit memory allocation
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

BASE_DIR = r'D:\AI\Computer Vision\projects\face_identifier'
MODEL_DIR = os.path.join(BASE_DIR, 'model')
model_name = 'face_recognition_triplot.h5'
model_path = os.path.join(MODEL_DIR, model_name)
model = tf.keras.models.load_model(model_path, compile=False)



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

embedding_dataset = tf_load_embedding_dataset_for_deploy()

t1 = 0
t2 = 0

while cap.isOpened():
    isSuccess, img = cap.read()
    if isSuccess:
        boxes, probs = mtcnn.detect(img)
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                bbox = list(map(int,box.tolist()))
                face, status = extract_face(bbox, img, img.shape)
                if status == True:
                    user, score = tf_inference(face, model, embedding_dataset, device)
                if user == -1:
                    user = 'unknown'
                    img = cv2.putText(img, f'{user}', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.3, (0, 0, 255), 1, cv2.LINE_AA)
                    img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),1)
                else:
                    # Using cv2.putText() method to display score
                    #score = torch.round(score, decimals=4)
                    img = cv2.putText(img, f'{user}: {score}', (bbox[0],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.3, (0, 0, 255), 1, cv2.LINE_AA)
                    img = cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),1)
        else:
            print('Not detected face!')
    
    t2 = time.time()
    fps = int(1 / (t2 - t1))
    t1 = t2
    
    # Put fps on frame
    img = cv2.putText(img, f'{fps}', (0, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(1)&0xFF == 27:
        break