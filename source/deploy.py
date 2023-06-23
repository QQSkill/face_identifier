import cv2
from utils import extract_face,load_embedding_dataset_for_deploy, inference
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# Check type of device (cpu, gpu, tpu)
device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)

# Initalize face detector - MTCNN
mtcnn = MTCNN(margin = 20, keep_all=False, post_process=False, device = device)

# Intialize embedding model
model = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
).to(device)
model.eval()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

embedding_dataset = load_embedding_dataset_for_deploy()

while cap.isOpened():
    isSuccess, frame = cap.read()
    if isSuccess:
        boxes, probs = mtcnn.detect(img)
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                bbox = list(map(int,box.tolist()))
                face, status = extract_face(bbox, img, img.shape)
                if status == True:
                    user, score = inference(face, model, embedding_dataset)
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
            print('face detection is error')

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break