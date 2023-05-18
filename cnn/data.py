import os

import cv2
import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
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