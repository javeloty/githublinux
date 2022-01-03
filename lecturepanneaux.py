import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2

size = 42
dir_images_panneaux = "images_panneaux"
dir_images_autres_panneaux = "images_autres_panneaux"
dir_images_sans_panneaux = "images_sans_panneaux"


def panneau_model(nbr_classes):
    model = tf.keras.Sequential()

    model.add(layers.Input(shape=(size, size, 3), dtype='float32'))

    model.add(layers.Conv2D(128, 3, strides=1))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128, 3, strides=1))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Conv2D(256, 3, strides=1))
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, 3, strides=1))
    model.add(layers.Dropout(0.4))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(nbr_classes, activation='softmax'))

    return model


def is_panneau_model():
    model = tf.keras.Sequential()

    model.add(layers.Input(shape=(size, size, 3), dtype='float32'))

    model.add(layers.Conv2D(64, 3, strides=1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Conv2D(128, 3, strides=1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPool2D(pool_size=2, strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def lire_images_panneaux(dir_images_panneaux, size=None):
    tab_panneau = []
    tab_image_panneau = []

    if not os.path.exists(dir_images_panneaux):
        quit("Le repertoire d'image n'existe pas: {}".format(dir_images_panneaux))

    files = os.listdir(dir_images_panneaux)
    if files is None:
        quit("Le repertoire d'image est vide: {}".format(dir_images_panneaux))

    for file in sorted(files):
        if file.endswith("png"):
            tab_panneau.append(file.split(".")[0])
            image = cv2.imread(dir_images_panneaux + "/" + file)
            if size is not None:
                image = cv2.resize(image, (size, size), cv2.INTER_LANCZOS4)
            tab_image_panneau.append(image)

    return tab_panneau,

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@data set
import numpy as np
import cv2
from multiprocessing import Pool
import multiprocessing
import random


def bruit(image_orig):
    h, w, c = image_orig.shape
    n = np.random.randn(h, w, c) * random.randint(5, 30)
    return np.clip(image_orig + n, 0, 255).astype(np.uint8)


def change_gamma(image, alpha=1.0, beta=0.0):
    return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)


def modif_img(img):
    h, w, c = img.shape

    r_color = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
    img = np.where(img == [142, 142, 142], r_color, img).astype(np.uint8)

    if np.random.randint(3):
        k_max = 3
        kernel_blur = np.random.randint(k_max) * 2 + 1
        img = cv2.GaussianBlur(img, (kernel_blur, kernel_blur), 0)

    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), random.randint(-10, 10), 1)
    img = cv2.warpAffine(img, M, (w, h))

    if np.random.randint(2):
        a = int(max(w, h) / 5) + 1
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[0 + random.randint(-a, a), 0 + random.randint(-a, a)],
                           [w - random.randint(-a, a), 0 + random.randint(-a, a)],
                           [0 + random.randint(-a, a), h - random.randint(-a, a)],
                           [w - random.randint(-a, a), h - random.randint(-a, a)]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h))

    if np.random.randint(2):
        r = random.randint(0, 5)
        h2 = int(h * 0.9)
        w2 = int(w * 0.9)
        if r == 0:
            img = img[0:w2, 0:h2]
        elif r == 1:
            img = img[w - w2:w, 0:h2]
        elif r == 2:
            img = img[0:w2, h - h2:h]
        elif r == 3:
            img = img[w - w2:w, h - h2:h]
        img = cv2.resize(img, (h, w))

    if np.random.randint(2):
        r = random.randint(1, int(max(w, h) * 0.15))
        img = img[r:w - r, r:h - r]
        img = cv2.resize(img, (h, w))

    if not np.random.randint(4):
        t = np.empty((h, w, c), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    t[i][j][k] = (i / h)
        M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), np.random.randint(4) * 90, 1)
        t = cv2.warpAffine(t, M, (w, h))
        img = (cv2.multiply((img / 255).astype(np.float32), t) * 255).astype(np.uint8)

    img = change_gamma(img, random.uniform(0.6, 1.0), -np.random.randint(50))

    if not np.random.randint(4):
        p = (15 + np.random.randint(10)) / 100
        img = (img * p + 50 * (1 - p)).astype(np.uint8) + np.random.randint(100)

    img = bruit(img)

    return img


def create_lot_img(image, nbr, nbr_thread=None):
    if nbr_thread is None:
        nbr_thread = multiprocessing.cpu_count()
    lot_original = np.repeat([image], nbr, axis=0)
    with Pool(nbr_thread) as p:
        lot_result = p.map(modif_img, lot_original)
        p.close()
    return lot_result
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@gener_fond@@@@@@@@@@@@@@@2222


import cv2
import numpy as np
import random
import os
import common

video = "videos/France_Motorway.mp4"

if not os.path.isdir(common.dir_images_sans_panneaux):
    os.mkdir(common.dir_images_sans_panneaux)

if not os.path.exists(video):
    print("Vidéo non présente:", video)
    quit()

cap = cv2.VideoCapture(video)

id = 0
nbr_image = 100000

nbr_image_par_frame = int(100000 / cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1

while True:
    ret, frame = cap.read()
    if ret is False:
        quit()
    h, w, c = frame.shape

    for cpt in range(nbr_image_par_frame):
        x = random.randint(0, w - common.size)
        y = random.randint(0, h - common.size)
        img = frame[y:y + common.size, x:x + common.size]
        cv2.imwrite(common.dir_images_sans_panneaux + "/{:d}.png".format(id), img)
        id += 1
        if id == nbr_image:
            quit()

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@gener_panneaux@@@@@@@@@@@
import numpy as np
from sklearn.utils import shuffle
import cv2
import common
import dataset

tab_panneau, tab_image_panneau=common.lire_images_panneaux(common.dir_images_panneaux, common.size)

tab_images=np.array([]).reshape(0, common.size, common.size, 3)
tab_labels=[]

id=0
for image in tab_image_panneau:
    lot=dataset.create_lot_img(image, 1000)
    tab_images=np.concatenate([tab_images, lot])
    tab_labels=np.concatenate([tab_labels, np.full(len(lot), id)])
    id+=1

tab_panneau=np.array(tab_panneau)
tab_images=np.array(tab_images, dtype=np.float32)/255
tab_labels=np.array(tab_labels).reshape([-1, 1])

tab_images, tab_labels=shuffle(tab_images, tab_labels)

for i in range(len(tab_images)):
    cv2.imshow("panneau", tab_images[i])
    print("label", tab_labels[i], "panneau", tab_panneau[int(tab_labels[i])])
    if cv2.waitKey()&0xFF==ord('q'):
        quit()
#"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@highcircle
import cv2
import numpy as np

param1=30
param2=55
dp=1.0

cap=cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles=cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, 20, param1=param1, param2=param2, minRadius=10, maxRadius=50)
    if circles is not None:
        circles=np.around(circles).astype(np.int32)
        for i in circles[0, :]:
            if i[2]!=0:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 4)
    cv2.putText(frame, "[i|k]dp: {:4.2f}  [o|l]param1: {:d}  [p|m]param2: {:d}".format(dp, param1, param2), (10, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
    cv2.imshow("Video", frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        quit()
    if key==ord('i'):
        dp=min(10, dp+0.1)
    if key==ord('k'):
        dp=max(0.1, dp-0.1)
    if key==ord('o'):
        param1=min(255, param1+1)
    if key==ord('l'):
        param1=max(1, param1-1)
    if key==ord('p'):
        param2=min(255, param2+1)
    if key==ord('m'):
        param2=max(1, param2-1)
cv2.destroyAllWindows()

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@lecture de panneaux
import tensorflow as tf
import cv2
import os
import numpy as np
import random
import common

th1 = 30
th2 = 55

video_dir = "dashcam Cedric"

tab_panneau, tab_image_panneau = common.lire_images_panneaux(common.dir_images_panneaux)

model_is_panneau = common.is_panneau_model()
checkpoint = tf.train.Checkpoint(model_is_panneau=model_is_panneau)
checkpoint.restore(tf.train.latest_checkpoint("./training_is_panneau/"))

model_panneau = common.panneau_model(len(tab_panneau))
checkpoint = tf.train.Checkpoint(model_panneau=model_panneau)
checkpoint.restore(tf.train.latest_checkpoint("./training_panneau/"))

l = os.listdir(video_dir)
random.shuffle(l)

for video in l:
    if not video.endswith("mp4"):
        continue
    cap = cv2.VideoCapture(video_dir + "/" + video)

    print("video:", video)
    id_panneau = -1
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        f_w, f_h, f_c = frame.shape
        frame = cv2.resize(frame, (int(f_h / 1.5), int(f_w / 1.5)))

        image = frame[200:400, 700:1000]
        cv2.rectangle(frame, (700, 200), (1000, 400), (255, 255, 255), 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=th1, param2=th2, minRadius=5, maxRadius=45)
        if circles is not None:
            circles = np.int16(np.around(circles))
            for i in circles[0, :]:
                if i[2] != 0:
                    panneau = cv2.resize(image[max(0, i[1] - i[2]):i[1] + i[2], max(0, i[0] - i[2]):i[0] + i[2]],
                                         (common.size, common.size)) / 255
                    cv2.imshow("panneau", panneau)
                    prediction = model_is_panneau(np.array([panneau]), training=False)
                    print("prediction", prediction)
                    if prediction[0][0] > 0.9:
                        prediction = model_panneau(np.array([panneau]), training=False)
                        id_panneau = np.argmax(prediction[0])
                        print("panneau", prediction, id_panneau, tab_panneau[id_panneau])
                        w, h, c = tab_image_panneau[id_panneau].shape
        if id_panneau != -1:
            frame[0:h, 0:w, :] = tab_image_panneau[id_panneau]
        cv2.putText(frame, "fichier:" + video, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit()
        if key == ord('a'):
            for cpt in range(100):
                ret, frame = cap.read()
        if key == ord('f'):
            break

cv2.destroyAllWindows()

#"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@trian is panneau@@@@@@@@@@@@@


import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import os
import common
import time
import dataset

batch_size = 64
nbr_entrainement = 20

tab_images = np.array([]).reshape(0, common.size, common.size, 3)

tab_panneau, tab_image_panneau = common.lire_images_panneaux(common.dir_images_panneaux, common.size)

if not os.path.exists(common.dir_images_autres_panneaux):
    quit("Le repertoire d'image n'existe pas: {}".format(common.dir_images_autres_panneaux))

if not os.path.exists(common.dir_images_sans_panneaux):
    quit("Le repertoire d'image n'existe pas:".format(common.dir_images_sans_panneaux))

nbr = 0
for image in tab_image_panneau:
    lot = dataset.create_lot_img(image, 12000)
    tab_images = np.concatenate([tab_images, lot])
    nbr += len(lot)

tab_labels = np.full(nbr, 1)

print("Image panneaux:", nbr)

files = os.listdir(common.dir_images_autres_panneaux)
if files is None:
    quit("Le repertoire d'image est vide:".format(common.dir_images_autres_panneaux))

nbr = 0
for file in files:
    if file.endswith("png"):
        path = os.path.join(common.dir_images_autres_panneaux, file)
        image = cv2.resize(cv2.imread(path), (common.size, common.size), cv2.INTER_LANCZOS4)
        lot = dataset.create_lot_img(image, 700)
        tab_images = np.concatenate([tab_images, lot])
        nbr += len(lot)

tab_labels = np.concatenate([tab_labels, np.full(nbr, 0)])

print("Image autres panneaux:", nbr)

nbr_np = int(len(tab_images) / 2)
print("nbr_np", nbr_np)

id = 1
nbr = 0
tab = []
for cpt in range(nbr_np):
    file = common.dir_images_sans_panneaux + "/{:d}.png".format(id)
    if not os.path.isfile(file):
        break
    image = cv2.resize(cv2.imread(file), (common.size, common.size))
    tab.append(image)
    id += 1
    nbr += 1

tab_images = np.concatenate([tab_images, tab])
tab_labels = np.concatenate([tab_labels, np.full(nbr, 0)])
print("Image sans panneaux:", nbr)

tab_images = np.array(tab_images, dtype=np.float32) / 255
tab_labels = np.array(tab_labels, dtype=np.float32).reshape([-1, 1])

tab_images, tab_labels = shuffle(tab_images, tab_labels)
train_images, test_images, train_labels, test_labels = train_test_split(tab_images, tab_labels, test_size=0.10)

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

print("train_images", len(train_images))
print("test_images", len(test_images))
print("nbr panneau", len(np.where(train_labels == 0.)[1]), train_labels.shape)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model_is_panneau(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model_is_panneau.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_is_panneau.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


def train(train_ds, nbr_entrainement):
    for entrainement in range(nbr_entrainement):
        start = time.time()
        for images, labels in train_ds:
            train_step(images, labels)
        message = 'Entrainement {:04d}, loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
        print(message.format(entrainement + 1,
                             train_loss.result(),
                             train_accuracy.result() * 100,
                             time.time() - start))
        train_loss.reset_states()
        train_accuracy.reset_states()
        test(test_ds)


def test(test_ds):
    start = time.time()
    for test_images, test_labels in test_ds:
        predictions = model_is_panneau(test_images)
        t_loss = loss_object(test_labels, predictions)
        test_loss(t_loss)
        test_accuracy(test_labels, predictions)
    message = '   >>> Test: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
    print(message.format(test_loss.result(),
                         test_accuracy.result() * 100,
                         time.time() - start))
    test_loss.reset_states()
    test_accuracy.reset_states()


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.BinaryCrossentropy()
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.BinaryAccuracy()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.BinaryAccuracy()
model_is_panneau = common.is_panneau_model()
checkpoint = tf.train.Checkpoint(model_is_panneau=model_is_panneau)

print("Entrainement")
train(train_ds, nbr_entrainement)
test(test_ds)

checkpoint.save(file_prefix="./training_is_panneau/is_panneau")

#@@@@@@@@@@@@@@@@@@@@@@@@@@@trainispanneau@@@@@@@@@@@@@@@@@@@@@@@22


import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import os
import time
import common
import dataset

batch_size = 128
nbr_entrainement = 20

tab_images = np.array([]).reshape(0, common.size, common.size, 3)
tab_labels = []

tab_panneau, tab_image_panneau = common.lire_images_panneaux(common.dir_images_panneaux, common.size)

id = 0
for image in tab_image_panneau:
    lot = dataset.create_lot_img(image, 12000)
    tab_images = np.concatenate((tab_images, lot))
    tab_labels = np.concatenate([tab_labels, np.full(len(lot), id)])
    id += 1

tab_panneau = np.array(tab_panneau)
tab_images = np.array(tab_images, dtype=np.float32) / 255
tab_labels = np.array(tab_labels, dtype=np.float32).reshape([-1, 1])

train_images, test_images, train_labels, test_labels = train_test_split(tab_images, tab_labels, test_size=0.10)

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

print("train_images", len(train_images))
print("test_images", len(test_images))


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model_panneau(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model_panneau.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_panneau.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


def train(train_ds, nbr_entrainement):
    for entrainement in range(nbr_entrainement):
        start = time.time()
        for images, labels in train_ds:
            train_step(images, labels)
        message = 'Entrainement {:04d}: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
        print(message.format(entrainement + 1,
                             train_loss.result(),
                             train_accuracy.result() * 100,
                             time.time() - start))
        train_loss.reset_states()
        train_accuracy.reset_states()
        test(test_ds)


def test(test_ds):
    start = time.time()
    for test_images, test_labels in test_ds:
        predictions = model_panneau(test_images)
        t_loss = loss_object(test_labels, predictions)
        test_loss(t_loss)
        test_accuracy(test_labels, predictions)
    message = '   >>> Test: loss: {:6.4f}, accuracy: {:7.4f}%, temps: {:7.4f}'
    print(message.format(test_loss.result(),
                         test_accuracy.result() * 100,
                         time.time() - start))
    test_loss.reset_states()
    test_accuracy.reset_states()


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
model_panneau = common.panneau_model(len(tab_panneau))
checkpoint = tf.train.Checkpoint(model_panneau=model_panneau)

print("Entrainement")
train(train_ds, nbr_entrainement)
checkpoint.save(file_prefix="./training_panneau/panneau")

for i in range(len(test_images)):
    prediction = model_panneau(np.array([test_images[i]]))
    print("prediction", prediction, tab_panneau[np.argmax(prediction[0])])
    cv2.imshow("image", test_images[i])
    if cv2.waitKey() & 0xFF == ord('q'):
        break

