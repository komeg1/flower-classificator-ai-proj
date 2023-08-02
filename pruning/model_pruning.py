import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import CategoricalCrossentropy, KLDivergence, SparseCategoricalCrossentropy
from keras.models import load_model
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt

import cnn.definitions as defs
import cnn.data as data
import tensorflow as tf
import numpy as np
from numpy import linalg as LA


tf.random.set_seed(47)

model = load_model('../models/cnn_model_7k.h5')

model.summary()
x_train, x_test, y_train, y_test = data.prepare_data3()
model.evaluate(x_test, y_test)



def prune_dense(k_weights, b_weights, k_sparsity):
    kernel_weights = np.copy(k_weights)

    sorted_indices = np.argsort(np.abs(kernel_weights), axis=None)
    unraveled_indices = np.unravel_index(sorted_indices, kernel_weights.shape)


    cut = int(k_sparsity*len(unraveled_indices[0]))
    sparse_cut_inds = (unraveled_indices[0][0:cut], unraveled_indices[1][0:cut])
    kernel_weights[sparse_cut_inds] = 0.

    bias_weights = np.copy(b_weights)
    abs_bias_weights = np.abs(bias_weights)
    sorted_indices = np.argsort(abs_bias_weights, axis=None)
    unraveled_indices = np.unravel_index(sorted_indices, bias_weights.shape)


    cut = int(len(unraveled_indices[0]) * k_sparsity)
    print('cutoff: ', cut)
    sparse_cut_inds = (unraveled_indices[0][0:cut])
    bias_weights[sparse_cut_inds] = 0.

    return kernel_weights, bias_weights


#procent usunięcia wag
k_sparsities = []
for i in range(0, 100, 5):
    sparsity = i / 100
    k_sparsities.append(sparsity)



def sparsify_model(model, k_sparsity):

    sparse_model = tf.keras.models.clone_model(model)
    sparse_model.set_weights(model.get_weights())

    weights = sparse_model.get_weights()

    newWeightList = []

    for i in range(0, len(weights) - 4):
        unmodified_weight = np.copy(weights[i])
        newWeightList.append(unmodified_weight)

    for i in range(len(weights)-4, len(weights), 2):
        kernel_weights, bias_weights = prune_dense(weights[i], weights[i + 1], k_sparsity)
        newWeightList.append(kernel_weights)
        newWeightList.append(bias_weights)

    sparse_model.set_weights(newWeightList)

    sparse_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    score = sparse_model.evaluate(x_test, y_test, verbose=0)
    print('k% weight sparsity: ', k_sparsity,
          '\tTest loss: {:07.5f}'.format(score[0]),
          '\tTest accuracy: {:05.2f} %%'.format(score[1]*100))

    return sparse_model, score


model_loss_weight = []
model_accs_weight = []
kernel_weights_list = []
bias_weights_list = []
zero_weights_count_list = []

k_spars_optimized = 0.6

sparse_model, score = sparsify_model(model, k_spars_optimized)
sparse_model.fit(x_train, y_train, epochs=5, batch_size=defs.BATCH_SIZE, validation_data= (x_test, y_test))
#sparse_model.save('sparse_model_7_classes.h5')



