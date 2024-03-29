import os

import cv2
import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random as rn

def prepare_data():
    X=[]
    Z=[]
    IMG_SIZE=150
    FLOWER_DAISY_DIR='../flowers/daisy'

    FLOWER_SUNFLOWER_DIR='../flowers/sunflower'
    FLOWER_TULIP_DIR='../flowers/tulip'
    FLOWER_DANDI_DIR='../flowers/dandelion'
    FLOWER_ROSE_DIR='../flowers/rose'

    def assign_label(img,flower_type):
        return flower_type


    def make_train_data(flower_type, DIR):
        for img in tqdm(os.listdir(DIR)):
            label = assign_label(img, flower_type)

            path = os.path.join(DIR, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            X.append(np.array(img))
            Z.append(str(label))

    make_train_data('Daisy',FLOWER_DAISY_DIR)
    print(len(X))

    make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
    print(len(X))

    make_train_data('Tulip',FLOWER_TULIP_DIR)
    print(len(X))

    make_train_data('Dandelion',FLOWER_DANDI_DIR)
    print(len(X))

    make_train_data('Rose',FLOWER_ROSE_DIR)
    print(len(X))


    le=LabelEncoder()
    Y=le.fit_transform(Z)
    Y=to_categorical(Y,5)
    X=np.array(X)
    X=X/255

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

    return x_train,x_test,y_train,y_test


def prepare_data2():
    X = []
    Z = []
    IMG_SIZE = 150

    FLOWER_ASTILBE_DIR = '../flowers2/astilbe'
    FLOWER_BELLFLOWER_DIR = '../flowers2/bellflower'
    FLOWER_BLACKEYED_DIR = '../flowers2/black_eyed_susan'
    FLOWER_CALENDULA_DIR = '../flowers2/calendula'
    FLOWER_CALIFORNIA_DIR = '../flowers2/california_poppy'
    FLOWER_CARNATION_DIR = '../flowers2/carnation'
    FLOWER_COMMONDAISY_DIR = '../flowers2/common_daisy'
    FLOWER_COREOPSIS_DIR = '../flowers2/coreopsis'
    FLOWER_DAFFODIL_DIR = '../flowers2/daffodil'
    FLOWER_DANDELION_DIR = '../flowers2/dandelion'
    FLOWER_IRIS_DIR = '../flowers2/iris'
    FLOWER_MAGNOLIA_DIR = '../flowers2/magnolia'
    FLOWER_ROSE_DIR = '../flowers2/rose'
    FLOWER_SUNFLOWER_DIR = '../flowers2/sunflower'
    FLOWER_TULIP_DIR = '../flowers2/tulip'
    FLOWER_WATERLILY_DIR = '../flowers2/water_lily'

    def assign_label(img, flower_type):
        return flower_type

    def make_train_data(flower_type, DIR):
        for img in tqdm(os.listdir(DIR)):
            label = assign_label(img, flower_type)

            path = os.path.join(DIR, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            X.append(np.array(img))
            Z.append(str(label))

    make_train_data('Astible', FLOWER_ASTILBE_DIR)
    print(len(X))

    make_train_data('BellFlower', FLOWER_BELLFLOWER_DIR)
    print(len(X))

    make_train_data('BlackEyedSusan', FLOWER_BLACKEYED_DIR)
    print(len(X))

    make_train_data('Calendula', FLOWER_CALENDULA_DIR)
    print(len(X))

    make_train_data('CaliforniaPoppy', FLOWER_CALIFORNIA_DIR)
    print(len(X))

    make_train_data('Carnation', FLOWER_CARNATION_DIR)
    print(len(X))

    make_train_data('CommonDaisy', FLOWER_COMMONDAISY_DIR)
    print(len(X))

    make_train_data('Coreopsis', FLOWER_COREOPSIS_DIR)
    print(len(X))

    make_train_data('Daffodil', FLOWER_DAFFODIL_DIR)
    print(len(X))

    make_train_data('Dandelion', FLOWER_DANDELION_DIR)
    print(len(X))

    make_train_data('Iris', FLOWER_IRIS_DIR)
    print(len(X))

    make_train_data('Magnolia', FLOWER_MAGNOLIA_DIR)
    print(len(X))

    make_train_data('Rose', FLOWER_ROSE_DIR)
    print(len(X))

    make_train_data('Sunflower', FLOWER_SUNFLOWER_DIR)
    print(len(X))

    make_train_data('Tulip', FLOWER_TULIP_DIR)
    print(len(X))

    make_train_data('WaterLily', FLOWER_WATERLILY_DIR)
    print(len(X))

    le = LabelEncoder()
    Y = le.fit_transform(Z)
    Y = to_categorical(Y, 16)
    X = np.array(X)
    X = X / 255

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)


    return x_train, x_test, y_train, y_test


def prepare_data3():
    X = []
    Z = []
    IMG_SIZE = 150

    FLOWER_BELLFLOWER_DIR ='../flowers3/bellflower'
    FLOWER_DAISY_DIR = '../flowers3/daisy'
    FLOWER_DANDELION_DIR = '../flowers3/dandelion'
    FLOWER_LOTUS_DIR = '../flowers3/lotus'
    FLOWER_ROSE_DIR = '../flowers3/rose'
    FLOWER_SUNFLOWER_DIR = '../flowers3/sunflower'
    FLOWER_TULIP_DIR = '../flowers3/tulip'


    def assign_label(img,flower_type):
        return flower_type


    def make_train_data(flower_type, DIR):
        for img in tqdm(os.listdir(DIR)):
            label = assign_label(img, flower_type)

            path = os.path.join(DIR, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            try:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            except:
                print(path)
                continue

            X.append(np.array(img))
            Z.append(str(label))

    make_train_data('BellFlower', FLOWER_BELLFLOWER_DIR)
    print(len(X))

    make_train_data('Daisy', FLOWER_DAISY_DIR)
    print(len(X))

    make_train_data('Dandelion', FLOWER_DANDELION_DIR)
    print(len(X))

    make_train_data('Lotus', FLOWER_LOTUS_DIR)
    print(len(X))

    make_train_data('Rose', FLOWER_ROSE_DIR)
    print(len(X))

    make_train_data('Sunflower', FLOWER_SUNFLOWER_DIR)
    print(len(X))

    make_train_data('Tulip', FLOWER_TULIP_DIR)
    print(len(X))

    le = LabelEncoder()
    Y = le.fit_transform(Z)
    Y = to_categorical(Y, 7)
    X = np.array(X)
    X = X / 255

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,random_state=42)


    return x_train, x_test, y_train, y_test