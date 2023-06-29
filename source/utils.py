import cv2
from facenet_pytorch import MTCNN
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf

BASE_DIR = r'D:\AI\Computer Vision\projects\face_identifier'
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

def extract_and_save_faces(dataset_dir, model, device):
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
  extract_and_save_faces(TRAIN_DATASET_DIR, face_detector, device)
  embedding_and_save(TRAIN_DATASET_DIR, face_embedder, device)
  print('DONE TRAIN preparing data step')

def prepare_data_for_testing(face_detector, face_embedder, device):
  print('PROCESSING TEST preparing data step')
  extract_and_save_faces(TEST_DATASET_DIR, face_detector, device)
  embedding_and_save(TEST_DATASET_DIR, face_embedder, device)
  print('DONE TEST preparing data step')


def load_embedding_dataset_for_deploy():
  result = {'embedding': [], 'user_name': []}
  for user in os.listdir(TRAIN_DATASET_DIR):
      EMBEDDING_FILE = os.path.join(TRAIN_DATASET_DIR, user, EMBEDDING_FOLDER, f'{user}.pth')
      embedding = torch.load(EMBEDDING_FILE)
      user_name = str(user)
      result['embedding'].append(embedding)
      result['user_name'].append(user_name)
  return result

def inference(face, model, embedding_dataset, device, threshold=0.8):
  local_embeds = torch.cat(embedding_dataset['embedding']).to(device)
  names = embedding_dataset['user_name']
  face = transfrom_img(face).to(device)
  face = face.unsqueeze(0)
  embed = model(face)
  # Euclid distance
  norm_diff = embed - local_embeds
  norm_square = torch.pow(norm_diff, 2)
  norm_score = torch.sum(norm_square, dim=1) #(1,n)
  #norm_score = torch.sqrt(norm_score)
  embed_idx = torch.argmin(norm_score)
  min_dist = norm_score[embed_idx]
  if min_dist > threshold:
    return -1, -1
  else:
    return names[embed_idx], min_dist

def load_dataset_for_recognition():
  training_faces, testing_faces, training_labels, testing_labels = [], [], [], []
  for user in os.listdir(TRAIN_DATASET_DIR):
    faces_folder = os.path.join(TRAIN_DATASET_DIR, user, FACE_FOLDER)
    for face in os.listdir(faces_folder):
      face_path = os.path.join(faces_folder, face)
      face = cv2.imread(face_path)
      #for trainign model - not pretrained model
      #face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
      training_faces.append(face)
      training_labels.append(user)
  
  for user in os.listdir(TEST_DATASET_DIR):
    faces_folder = os.path.join(TEST_DATASET_DIR, user, FACE_FOLDER)
    for face in os.listdir(faces_folder):
      face_path = os.path.join(faces_folder, face)
      face = cv2.imread(face_path)
      #for trainign model - not pretrained model
      #face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
      testing_faces.append(face)
      testing_labels.append(user)
  return training_faces, testing_faces, training_labels, testing_labels

def recognition_evaluation(model):
  embedding_dataset = load_embedding_dataset_for_deploy()
  training_faces, testing_faces, training_labels, testing_labels = load_dataset_for_recognition()
  predicts = []
  for face in testing_faces:
    user, score = inference(face, model, embedding_dataset)
    if user == -1:
      user = 'unknown'
    else:
      # Using cv2.putText() method to display score
      score = torch.round(score, decimals=4)
    predicts.append(user)
  acc = accuracy_score(testing_labels, predicts)
  print('accuracy: ', acc)

def tf_embedding_and_save(dataset_dir, model, device):
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
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = np.stack([img])
        embed = model.predict(img)
        embeds.append(embed) #1 anh, kich thuoc [1,512]
        if len(embeds) == 0:
            continue
    embedding = tf.concat(embeds, 0, name='concat')
    embedding = tf.reduce_mean(embedding, 0)
    embedding_path = os.path.join(embedding_dir, f'{folder}.pth')
    with open(embedding_path, 'wb') as file:
      # A new file will be created
      pickle.dump(embedding, file)
  
def tf_prepare_data_for_training(face_detector, face_embedder, device):
  print('PROCESSING TRAIN preparing data step')
  #training dataset
  extract_and_save_faces(TRAIN_DATASET_DIR, face_detector, device)
  tf_embedding_and_save(TRAIN_DATASET_DIR, face_embedder, device)
  print('DONE TRAIN preparing data step')

def tf_prepare_data_for_testing(face_detector, face_embedder, device):
  print('PROCESSING TEST preparing data step')
  #testing dataset
  extract_and_save_faces(TEST_DATASET_DIR, face_detector, device)
  tf_embedding_and_save(TEST_DATASET_DIR, face_embedder, device)
  print('DONE TEST preparing data step')

def tf_load_embedding_dataset_for_deploy():
  result = {'embedding': [], 'user_name': []}
  for user in os.listdir(TRAIN_DATASET_DIR):
      EMBEDDING_FILE = os.path.join(TRAIN_DATASET_DIR, user, EMBEDDING_FOLDER, f'{user}.pth')
      with open(EMBEDDING_FILE, 'rb') as file:
        # Call load method to deserialze
        embedding = pickle.load(file)
      user_name = str(user)
      result['embedding'].append(embedding)
      result['user_name'].append(user_name)
  return result

def tf_load_dataset_for_recognition():
  training_faces, testing_faces, training_labels, testing_labels = [], [], [], []
  for user in os.listdir(TRAIN_DATASET_DIR):
    faces_folder = os.path.join(TRAIN_DATASET_DIR, user, FACE_FOLDER)
    for face in os.listdir(faces_folder):
      face_path = os.path.join(faces_folder, face)
      face = cv2.imread(face_path)
      #for trainign model - not pretrained model
      face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
      training_faces.append(face)
      training_labels.append(user)
  
  for user in os.listdir(TEST_DATASET_DIR):
    faces_folder = os.path.join(TEST_DATASET_DIR, user, FACE_FOLDER)
    for face in os.listdir(faces_folder):
      face_path = os.path.join(faces_folder, face)
      face = cv2.imread(face_path)
      #for trainign model - not pretrained model
      face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
      testing_faces.append(face)
      testing_labels.append(user)
  return training_faces, testing_faces, training_labels, testing_labels

def tf_inference(face, model, embedding_dataset, device, threshold=0.8, verbose=0):
  local_embeds = embedding_dataset['embedding']
  names = embedding_dataset['user_name']
  #print(face.shape, local_embeds.shape)
  face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
  face = np.stack([face])
  embed = model.predict(face, verbose=verbose)
  norm_diff = embed - local_embeds
  norm_square = np.power(norm_diff, 2)
  norm_score = np.sum(norm_square, axis=1) #(1,n)
  #norm_score = torch.sqrt(norm_score)
  embed_idx = np.argmin(norm_score)
  min_dist = norm_score[embed_idx]
  if min_dist > threshold:
      return -1, -1
  else:
      return names[embed_idx], min_dist

def tf_recognition_evaluation(model):
  embedding_dataset = tf_load_embedding_dataset_for_deploy()
  training_faces, testing_faces, training_labels, testing_labels = tf_load_dataset_for_recognition()
  predicts = []
  for face in testing_faces:
    user, score = tf_inference(face, model, embedding_dataset)
    if user == -1:
      user = 'unknown'
    else:
      # Using cv2.putText() method to display score
      #score = torch.round(score, decimals=4)
      pass
    predicts.append(user)
  print(testing_labels)
  print(predicts)
  acc = accuracy_score(testing_labels, predicts)
  print('accuracy: ', acc)