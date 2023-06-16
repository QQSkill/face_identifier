import cv2
from facenet_pytorch import MTCNN
import torch
import os
import numpy as np

BASE_DIR = '/content/drive/MyDrive/projects/ComputerVision/identifier_system_by_face'
TRAIN_DATASET_DIR = os.path.join(BASE_DIR, 'train_dataset')
TEST_DATASET_DIR = os.path.join(BASE_DIR, 'test_dataset')
IMG_FOLDER = 'img'
FACE_FOLDER = 'faces'
EMBEDDING_FOLDER = 'embeddings'

def extract_face(box, img, frame_size, margin=20):
    face_size = 160
    img_size = frame_size

    if box[0] > img_size[1] or box[2] > img_size[1]:
      print('Error face detection')
      return img, False

    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ]
    margin_box = [ #box[0] và box[1] là tọa độ của điểm góc trên cùng trái
        int(max(box[0] - margin[0] / 2, 0)), #nếu thêm vào margin bị ra khỏi rìa ảnh => đưa về điểm 0
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])), #nếu thêm vào margin bị ra khỏi rìa ảnh => đưa về tọa độ của ảnh gốc
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ] #tạo margin mới bao quanh box cũ
    try:
      margin_img = img[margin_box[1]:margin_box[3], margin_box[0]:margin_box[2], :]
      face = cv2.resize(margin_img,(face_size, face_size), interpolation=cv2.INTER_AREA)
    except:
      no_margin_img = img[box[1]:box[3], box[0]:box[2], :]
      face = cv2.resize(no_margin_img,(face_size, face_size), interpolation=cv2.INTER_AREA)


    #convert numpy array to image format
    #face = Img.fromarray(face)
    return face, True

def save_img_to_file(img, img_path):
  cv2.imwrite(img_path, img)

def transfrom_img(img):
  img = np.moveaxis(img, -1, 0)
  img = torch.from_numpy(img)
  return img/255

def extract_and_save_faces(dataset_dir, model):
  for folder in os.listdir(dataset_dir):
    img_dir = os.path.join(dataset_dir, folder, IMG_FOLDER)
    face_dir = os.path.join(dataset_dir, folder, FACE_FOLDER)
    if os.path.exists(face_dir) != True:
      os.mkdir(face_dir)
    for img_file in os.listdir(img_dir):
      img_path = os.path.join(img_dir, img_file)
      input_img = cv2.imread(img_path)
      boxes, _ = model.detect(input_img)
      if boxes is not None:
          for idx, box in enumerate(boxes):
              bbox = list(map(int,box.tolist()))
              face, status = extract_face(bbox, input_img, input_img.shape)
              if status == True:
                face_path = os.path.join(face_dir, img_file)
                try:
                  save_img_to_file(face, face_path)
                except:
                  img_file = f'{img_file.split(".")[0]}.jpg'
                  face_path = os.path.join(face_dir, img_file)
                  save_img_to_file(face, face_path)

def embedding_and_save(dataset_dir, model, device):
  # Embedding face and save it
  for folder in os.listdir(dataset_dir):
    face_dir = os.path.join(dataset_dir, folder, FACE_FOLDER)
    embedding_dir = os.path.join(dataset_dir, folder, EMBEDDING_FOLDER)
    if os.path.exists(embedding_dir) != True:
      os.mkdir(embedding_dir)
    embeds = []
    for face_file in os.listdir(face_dir):
        face_path = os.path.join(face_dir, face_file)
        try:
            img = cv2.imread(face_path)
        except:
            continue
        with torch.no_grad():
            img = transfrom_img(img).to(device)
            img = img.unsqueeze(0)
            embed = model(img)
            embeds.append(embed) #1 anh, kich thuoc [1,512]
        if len(embeds) == 0:
            continue
    embedding = torch.cat(embeds).mean(0, keepdim=True) #dua ra trung binh cua 50 anh, kich thuoc [1,512]
    embedding_path = os.path.join(embedding_dir, f'{folder}.pth')
    torch.save(embedding, embedding_path)

def prepare_data_for_training(face_detector, face_embedder, device):
  print('PROCESSING TRAIN preparing data step')
  extract_and_save_faces(TRAIN_DATASET_DIR, face_detector)
  embedding_and_save(TRAIN_DATASET_DIR, face_embedder, device)
  print('DONE TRAIN preparing data step')

def prepare_data_for_testing(face_detector, face_embedder, device):
  print('PROCESSING TEST preparing data step')
  extract_and_save_faces(TEST_DATASET_DIR, face_detector)
  embedding_and_save(TEST_DATASET_DIR, face_embedder, device)
  print('DONE TEST preparing data step')
