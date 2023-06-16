from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import tensorflow_addons as tfa

#check type of device (cpu, gpu, tpu)
def _base_network():
  model = VGG16(include_top = True, weights = None)
  dense = Dense(128)(model.layers[-4].output)
  norm2 = Lambda(lambda x: tf.math.l2_normalize(x, axis = 1))(dense)
  model = Model(inputs = [model.input], outputs = [norm2])
  return model

model = _base_network()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tfa.losses.TripletSemiHardLoss())

X_train, testing_faces, y_train, testing_labels = load_dataset_for_recognition()
gen_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(1024).batch(32)

history = model.fit(
    gen_train,
    steps_per_epoch = 50,
    epochs=5)

MODEL_DIR = os.path.join(BASE_DIR, 'model')
if os.path.exists(MODEL_DIR) != True:
    os.mkdir(MODEL_DIR)
model_name = 'face_recognition_triplot.h5'
model_path = os.path.join(MODEL_DIR, model_name)
model.save(model_path)