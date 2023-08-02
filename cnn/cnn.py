import keras
import tf as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import h5py
import data as data
import numpy as np
from keras.models import Sequential
import definitions as Param
import tensorflow as tf

x_train, x_test, y_train, y_test = data.prepare_data()
#x_train, x_test, y_train, y_test = data.prepare_data2()
#x_train, x_test, y_train, y_test = data.prepare_data3()

np.random.seed(50)
tf.random.set_seed(55)


model = Sequential()
model.add(Conv2D(filters=Param.CONV_FILTERS_SIZE,
                 kernel_size=Param.KERNEL_SIZE_5,
                 padding='Same',
                 activation='relu',
                 input_shape=Param.INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=Param.POOL_SIZE))
model.add(Conv2D(filters=Param.CONV_FILTERS_SIZE * 2, #64
                 kernel_size=Param.KERNEL_SIZE_3,
                 padding='Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=Param.POOL_SIZE))
model.add(Conv2D(filters=Param.CONV_FILTERS_SIZE * 3,#96
                 kernel_size=Param.KERNEL_SIZE_3,
                 padding='Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=Param.POOL_SIZE))
model.add(Conv2D(filters=Param.CONV_FILTERS_SIZE * 3,
                 kernel_size=Param.KERNEL_SIZE_3,
                 padding='Same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=Param.POOL_SIZE))

model.add(Flatten())
model.add(Dense(Param.DENSE_SIZE,
                activation="relu"))
model.add(Dense(Param.CLASS_CNT,
                activation="softmax"))




datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

modelHistory = model.fit(x_train,y_train,
                    batch_size=Param.BATCH_SIZE,
                    epochs = Param.EPOCHS,
                    validation_data = (x_test,y_test))

plt.plot(modelHistory.history['val_accuracy'])

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['CNN 5 classes'], loc='upper left')
plt.show()

plt.plot(modelHistory.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['CNN 5 classes'], loc='upper left')
plt.show()


model.save('cnn_model_7k.h5')






