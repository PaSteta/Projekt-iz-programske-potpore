import pickle

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from mpl_toolkits.mplot3d import Axes3D

import binvox_rw
import dataset

MODELS_ROOT = 'models'
img_path = './input-image/12.png'
model_path = f'./{MODELS_ROOT}/sequential-date20200123-181556-objNUM2-epochs30-batch32-valSplit0.3.model'
IMG_SIZE = 50

gpu_option = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)  # 70% gpu memorije za trening
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_option))
tf.compat.v1.keras.backend.set_session(sess)

file_id = model_path.split('-date')[1].split('-obj')[0]
CATEGORIES_PATH = f'./categories/{file_id}.pickle'

categories = pickle.load(open(CATEGORIES_PATH, "rb"))
categories = list(categories)

print("Loading model...")
model = models.load_model(model_path)
print("Loaded!")

def read_image():
    print("...loading image...")
    pic = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pic = cv2.resize(pic, (IMG_SIZE, IMG_SIZE))
    pic = pic.reshape((-1, IMG_SIZE, IMG_SIZE, 1))
    print("...loaded!")
    return pic

print("Prediction started")
prediction = model.predict(read_image())
print("Prediction finished!")

img_pred_num = prediction.argmax(axis=-1)[0]
print("categories len:", categories.__len__(),"| image prediction position:",img_pred_num)

img_pred = categories[img_pred_num]
print("prediction label:", img_pred)

vox_path = dataset.get_voxel_path(img_pred)
print("binvox path:",vox_path)

binv = open(vox_path, 'rb')
binvox_data = binvox_rw.read_as_3d_array(binv, False).data

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.voxels(binvox_data, edgecolors='k')
plt.show()
