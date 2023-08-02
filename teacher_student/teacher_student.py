import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.losses import CategoricalCrossentropy, KLDivergence, SparseCategoricalCrossentropy
from keras.models import load_model, clone_model
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import cnn.definitions as Param
import cnn.data as data
from Distiller import Distiller
import tensorflow as tf



tf.random.set_seed(1234)
#--- LOAD TEACHER MODEL ---#
teacher_model = load_model('../models/cnn_model_5k.h5')
teacher_model.summary()
x_train, x_test, y_train, y_test = data.prepare_data()

teacher_model.evaluate(x_test, y_test)



class MyCallback(keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_accuracy"]
        if val_acc > self.threshold:
            self.model.stop_training = True
            print(f'Reached on {epoch} epoch with val_acc {val_acc}')

student_model = Sequential()
student_model.add(Conv2D(filters=Param.CONV_FILTERS_SIZE,
              use_bias=True,
              kernel_size=Param.KERNEL_SIZE_3,
              input_shape=Param.INPUT_SHAPE))
student_model.add(Activation('relu'))
student_model.add(MaxPooling2D(pool_size=Param.POOL_SIZE))
student_model.add(Dropout(0.1))
student_model.add(Conv2D(filters=Param.CONV_FILTERS_SIZE*2,
                         use_bias=False,
                         kernel_size=Param.KERNEL_SIZE_3,
                         strides=3))
student_model.add(Activation('relu'))
student_model.add(MaxPooling2D(pool_size=Param.POOL_SIZE))
student_model.add(Dropout(0.2))
student_model.add(Conv2D(filters=Param.CONV_FILTERS_SIZE*2,
                         use_bias=False,
                         kernel_size=Param.KERNEL_SIZE_3,
                         strides=3))
student_model.add(Activation('relu'))
student_model.add(MaxPooling2D(pool_size=Param.POOL_SIZE))
student_model.add(Dropout(0.2))
student_model.add(Flatten())
student_model.add(Dense(Param.STUDENTS_DENSE_SIZE))
student_model.add(Activation('relu'))
student_model.add(Dropout(0.3))
student_model.add(Dense(Param.CLASS_CNT, activation="softmax"))

#--- COPY STUDENT MODEL ---#
student_copy = clone_model(student_model)
#-------------------------#
student_model.summary()

#--- DISTILLATION ---#
distiller = Distiller(student=student_model, teacher=teacher_model)
distiller.compile(
    optimizer=Adam(lr=0.001),
    metrics=['accuracy'],
    student_loss_fn=CategoricalCrossentropy(),
    distillation_loss_fn=KLDivergence(),
    alpha=Param.ALPHA,
    temperature=Param.TEMPERATURE
)

distillerHistory = distiller.fit(x_train, y_train,
                                 epochs=Param.STUDENTS_EPOCHS,
                                 batch_size=Param.STUDENTS_BATCH_SIZE,
                                 validation_data=(x_test, y_test),
                                 callbacks=[MyCallback(0.9)])


student_copy.compile( optimizer=Adam(lr=0.001),
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])

baseHistory = student_copy.fit(x_train, y_train,
                              epochs=Param.STUDENTS_EPOCHS,
                               batch_size=Param.STUDENTS_BATCH_SIZE,
                               validation_data=(x_test, y_test),
                               callbacks=[MyCallback(0.9)])



#student_copy = load_model('student_copy_5_classes.h5')
print("\nTeacher model:")
teacher_model.evaluate(x_test, y_test)

print("\nStudent model with knowledge distillation:")
distiller.evaluate(x_test, y_test)

print("\nStudent model without knowledge distillation:")
#student_copy.evaluate(x_test, y_test)

plt.plot(distillerHistory.history['val_accuracy'])
plt.plot(baseHistory.history['val_accuracy'])



