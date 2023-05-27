import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import CategoricalCrossentropy, KLDivergence, SparseCategoricalCrossentropy
from keras.models import load_model
from keras.optimizers import SGD

import cnn.definitions as defs
import cnn.data as data
from Distiller import Distiller
import tensorflow as tf

tf.random.set_seed(47)
#--- LOAD TEACHER MODEL ---#
teacher_model = load_model('cnn_model-v2.h5')
teacher_model.summary()
x_train, x_test, y_train, y_test = data.prepare_data()
teacher_model.evaluate(x_test, y_test)

#-------------------------#
#cnn_model = Sequential()
#cnn_model.add(Conv2D(filters=defs.CONV_FILTERS_SIZE, kernel_size=defs.KERNEL_SIZE, padding='Same', activation='relu', input_shape=defs.INPUT_SHAPE))
#cnn_model.add(MaxPooling2D(pool_size=defs.POOL_SIZE))
#cnn_model.add(Conv2D(filters=defs.CONV_FILTERS_SIZE*2, kernel_size=defs.KERNEL_SIZE, padding='Same', activation='relu'))
#cnn_model.add(MaxPooling2D(pool_size=defs.POOL_SIZE))
#
#cnn_model.add(Flatten())
#cnn_model.add(Dense(defs.DENSE_SIZE))
#cnn_model.add(Dense(defs.CLASS_CNT, activation="softmax"))

#--- CREATE STUDENT MODEL ---#
student_model = Sequential()
student_model.add(Conv2D(filters=defs.CONV_FILTERS_SIZE/4,
                         kernel_size=defs.KERNEL_SIZE,
                         padding='Same',
                         activation='relu',
                         input_shape=defs.INPUT_SHAPE))

student_model.add(MaxPooling2D(pool_size=defs.POOL_SIZE))

student_model.add(Conv2D(filters=defs.CONV_FILTERS_SIZE/2,
                         kernel_size=defs.KERNEL_SIZE,
                         padding='Same',
                         activation='relu'))

student_model.add(MaxPooling2D(pool_size=defs.POOL_SIZE))
student_model.add(Flatten())
student_model.add(Dense(defs.DENSE_SIZE/4))
student_model.add(Dense(defs.CLASS_CNT, activation="softmax"))
#---------------------------#

#--- COPY STUDENT MODEL ---#
student_model_copy = Sequential()
student_model_copy.add(Conv2D(filters=defs.CONV_FILTERS_SIZE/4,
                            kernel_size=defs.KERNEL_SIZE,
                            padding='Same',
                            activation='relu',
                            input_shape=defs.INPUT_SHAPE))

student_model_copy.add(MaxPooling2D(pool_size=defs.POOL_SIZE))

student_model_copy.add(Conv2D(filters=defs.CONV_FILTERS_SIZE/2,
                            kernel_size=defs.KERNEL_SIZE,
                            padding='Same',
                            activation='relu'))

student_model_copy.add(MaxPooling2D(pool_size=defs.POOL_SIZE))

student_model_copy.add(Flatten())
student_model_copy.add(Dense(defs.DENSE_SIZE/4))
student_model_copy.add(Dense(defs.CLASS_CNT, activation="softmax"))
#-------------------------#




student_model.summary()

#--- DISTILLATION ---#
distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer='adam',
    metrics=['accuracy'],
    student_loss_fn=CategoricalCrossentropy(),
    distillation_loss_fn=KLDivergence(),
    alpha=0.1,
    temperature=40,
)

distiller.fit(x_train, y_train, epochs=defs.EPOCHS, batch_size=defs.BATCH_SIZE, validation_data=(x_test, y_test))
#NIE MOGE ZPAISAC DO PLIKU BO OVERRIDE KLASY WAGI TEZ COS NIE DIZALAJA :(
#distiller.save_weights('distiller_model.h5')

student_model_copy.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
student_model_copy.fit(x_train, y_train, epochs=defs.EPOCHS, batch_size=defs.BATCH_SIZE, validation_data=(x_test, y_test))
student_model_copy.save('student_model_base.h5')

print("\nTeacher model:")
teacher_model.evaluate(x_test, y_test)

print("\nStudent model with knowledge distillation:")
distiller.evaluate(x_test, y_test)

print("\nStudent model without knowledge distillation:")
student_model_copy.evaluate(x_test, y_test)

