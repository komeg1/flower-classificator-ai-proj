from enum import Enum

from keras.models import load_model


class Model(Enum):
    REFERENCE =0,
    SIMPLE = 1,
    CNN = 2,
    MOD3 = 3,


CHOSEN_MODEL = Model.REFERENCE

modelFile = load_model('cnn_model.h5')

