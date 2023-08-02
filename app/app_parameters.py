from enum import Enum

from keras.models import load_model


class Model(Enum):
    REFERENCE =0,
    NO_DISTILLATION = 1,
    CNN = 2,
    DISTILLED = 3,
    PRUNED = 4,

class Dataset(Enum):
    FLOWERS_16 = 0,
    FLOWERS_5 = 1,
    FLOWERS_7 = 2,



