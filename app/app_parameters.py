from enum import Enum


class Model(Enum):
    REFERENCE =0,
    SIMPLE = 1,
    CNN = 2,
    MOD3 = 3,


CHOSEN_MODEL = Model.REFERENCE