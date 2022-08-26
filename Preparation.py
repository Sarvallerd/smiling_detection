import cv2
import numpy as np
import os

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


def cut(path: str):
    string = path
    list_image = []
    for i in os.listdir(string):
        real_path = string + '\\' + i
        if '.jpg' in real_path:
            image = cv2.imread(real_path, cv2.IMREAD_UNCHANGED)
            face_cascade = cv2.CascadeClassifier(
                r'D:\PycharmProjects\smiling_detection\haarcascade_frontalface_default.xml')
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
            a, b, c, d = 0, 0, 0, 0
            for (x, y, w, h) in faces:
                a = x
                b = y - 50
                c = x + w
                d = y + h + 30
            x, y, h, w = a, b, d - b, c - a
            corp_img = image[y:y + h, x: x + w]
            list_image.append(corp_img)
    return list_image


def get_array_from_image(path: str):
    l = cut(path)
    array = []
    for i in l:
        array.append(np.asarray(cv2.resize(i, (227, 227))))
    array_image = np.array(array)
    return array_image


def test_train_datasets(path: str):
    list_lables = []
    list_dir = []
    list_image = []
    for i in os.walk(path):
        for j in i:
            if 'test_folder\\1' in j:
                list_dir.append(j)
            elif 'test_folder\\0' in j:
                list_dir.append(j)
            elif 'train_folder\\0' in j:
                list_dir.append(j)
            elif 'train_folder\\1' in j:
                list_dir.append(j)
            break
    for i in list_dir:
        for j in os.listdir(i):
            a = cv2.imread(i + '\\' + j, cv2.IMREAD_UNCHANGED)
            list_image.append(cv2.resize(a, (227, 227)))
            if 'test_folder\\1' in i:
                list_lables.append(1)
            elif 'test_folder\\0' in i:
                list_lables.append(0)
            elif 'train_folder\\1' in i:
                list_lables.append(1)
            elif 'train_folder\\0' in i:
                list_lables.append(0)
    return shuffling(np.array(list_image), np.array(list_lables).reshape((len(list_lables), 1)))


def shuffling(array1, array2):
    randomize = np.arange(len(array1))
    np.random.shuffle(randomize)
    array1 = array1[randomize]
    array2 = array2[randomize]
    return array1, array2


def reformation(arr1, arr2):
    arr2 = np_utils.to_categorical(arr2, 2)
    arr1 = arr1.astype('float32')
    arr1 /= 255
    return arr1, arr2


def data_generation(path: str, batch_size: int, save_prefix: str, counter: int):
    image_list = []
    for i in os.listdir(path):
        img = cv2.imread(path + '\\' + i, cv2.IMREAD_UNCHANGED)
        image_list.append(img)
    image_array = np.array(image_list)
    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.1,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
    count = 0
    for batch in datagen.flow(image_array, batch_size=batch_size,
                              save_to_dir=path,
                              save_prefix=save_prefix, save_format='jpg'):
        count += 1
        if count > counter:
            break
