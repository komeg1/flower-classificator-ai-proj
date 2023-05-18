from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt

import data as data
import numpy as np
from keras.models import Sequential
import definitions as defs

x_train, x_test, y_train, y_test = data.prepare_data()

cnn_model = Sequential()
cnn_model.add(Conv2D(filters=defs.CONV_FILTERS_SIZE, kernel_size=defs.KERNEL_SIZE, padding='Same', activation='relu', input_shape=defs.INPUT_SHAPE))
cnn_model.add(MaxPooling2D(pool_size=defs.POOL_SIZE))
cnn_model.add(Conv2D(filters=defs.CONV_FILTERS_SIZE*2, kernel_size=defs.KERNEL_SIZE, padding='Same', activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=defs.POOL_SIZE))

cnn_model.add(Flatten())
cnn_model.add(Dense(defs.DENSE_SIZE))
cnn_model.add(Dense(defs.CLASS_CNT, activation="softmax"))

cnn_model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()


cnn_model.fit(x_train,y_train,epochs=defs.EPOCHS,batch_size=defs.BATCH_SIZE,validation_data = (x_test,y_test))

cnn_model.save('cnn_model.h5')




