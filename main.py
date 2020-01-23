import os
import pickle
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from tensorflow.keras.models import Sequential

import dataset

TRAINING_OBJECTS = ['Mobitel', 'Oruzje', 'Stolica', 'Stol', 'Auto',
                    'Brod', 'Lampa', 'Dvosjed', 'Klupa', 'Avion',
                    'Zvucnik', 'Ormar', 'Monitor']

curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
NAME = f"TENSORBOARD_PLOTS_{curr_time}"

CATEGORIES_ROOT = "categories"
MODELS_ROOT = 'models'

tensorboard = TensorBoard(
    log_dir=f'logs/{NAME}',
    write_graph=True,
    write_images=True
)

batch_size = 32
epochs = 30
IMG_SIZE = 50
valid_split = 0.3
OBJECT_NUM = 2
NUM_OF_IMAGES = 24
# -----------------


# GPU option
gpu_option = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)  # 70% gpu memorije za trening
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_option))
tf.compat.v1.keras.backend.set_session(sess)

# reading data...
data = dataset.get_train_data(OBJECT_NUM)

if len(data) == 0:
    print("Unijeli ste pogresan broj zeljenih objekata! Zatvaram ...")
    exit(1)

# random.shuffle(x_train)


if not os.path.exists(f'{MODELS_ROOT}'):
    os.mkdir(f'{MODELS_ROOT}')
if not os.path.exists(f'{CATEGORIES_ROOT}'):
    os.mkdir(f'{CATEGORIES_ROOT}')

# Getting categories from dataset and saving them for later use

category_path = f'./{CATEGORIES_ROOT}/{curr_time}'


def get_categories(data):
    c = set()
    for f, cat in data:
        c.add(cat)
    return list(c)


categories = get_categories(data)

with open(f'{category_path}.pickle', 'wb') as file:
    pickle.dump(categories, file)


# Iterate over the dataset to split the classes into training and testing data
def iterate_images(data):
    training = []
    test = []
    split = 0
    for img, class_key in data:
        i = categories.index(class_key)
        img_class_pair = [img, i]
        training.append(img_class_pair) if split < 20 \
            else test.append(img_class_pair)
        split += 1
        if split == NUM_OF_IMAGES:
            split = 0

    return training, test


training_data, test_data = iterate_images(data)

print(f'Trainging data is {len(training_data)}')
print(f'Test data is {len(test_data)}')


# Iterates over the given data to produce lists of features and labels
def iterate_data(data):
    X = []
    y = []
    for features, label in data:
        X.append(features)
        y.append(label)
    return X, y


X_train, y_train = iterate_data(training_data)
X_test, y_test = iterate_data(test_data)


def reshape_data(X, y):
    X = np.asarray(X).reshape((-1, IMG_SIZE, IMG_SIZE, 1))
    y = np.asarray(y)
    return X, y


X_train, y_train = reshape_data(X_train, y_train)
X_test, y_test = reshape_data(X_test, y_test)

# Normalizing the data for easier computation
X_train = X_train / 255.0
X_test = X_test / 255.0

# building layers
print('Building model...')
model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))

num = int(len(categories))
model.add(Dense(num))
model.add(Activation('softmax'))  # multiclass logistic regression, with K classes

# setting params
# adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)

TRAINING_OBJECTS = [i for i in TRAINING_OBJECTS[: OBJECT_NUM]]
print("===========", OBJECT_NUM, "OBJECTS TO TRAIN ON:", "; ".join(TRAINING_OBJECTS), "===========")

# training
model.fit(
    X_train,
    y_train,
    shuffle=True,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard]
)

model.save(
    f'./{MODELS_ROOT}/sequential-date{curr_time}-objNUM{OBJECT_NUM}-epochs{epochs}-batch{batch_size}-valSplit{valid_split}.model')
print("trained model id:", curr_time)

sess.close()
