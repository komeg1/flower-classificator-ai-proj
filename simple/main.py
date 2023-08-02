import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from simple.model_methods import *
import simple.data as data

#data.convert_to_csv()
data = data.load_data()

seed = np.random.seed(777)

data = np.array(data)
m,n = data.shape #label uwzgledniony
np.random.shuffle(data)

data_train = data[0:1000].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

#save the seed
IMG_SIZE = 128
PIXEL_SIZE = IMG_SIZE* IMG_SIZE
DENSE_SIZE = 5

def init_params():
    W1 = np.random.rand(DENSE_SIZE, IMG_SIZE) - 0.5
    b1 = np.random.rand(DENSE_SIZE, 1) - 0.5
    W2 = np.random.rand(DENSE_SIZE, 5) - 0.5
    b2 = np.random.rand(DENSE_SIZE, 1) - 0.5
    return W1, b1, W2, b2


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 1000)


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    #reflatten the image and plot the image
    current_image = current_image.reshape((128, 128))
    #resize to 128x128
    current_image = np.array(Image.fromarray(current_image).resize((128, 128)))
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    #add label and prediction to the plot
    plt.title("Prediction: " + str(prediction) + " Label: " + str(label))
    plt.imshow(current_image, cmap="hsv")
    plt.show()



    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)




test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)


train_predictions = make_predictions(X_train, W1, b1, W2, b2)
print(get_accuracy(train_predictions, Y_train))

save_model(W1, b1, W2, b2)




def test_image(W1, b1, W2, b2, image, label):

    img = Image.open(image).resize((128, 128)).convert("HSV")
    img = img.getdata(band=0)
    img = np.array(img)


    df = pd.DataFrame(img, index=label)
    df = np.array(df)
    df.T
    prediction = make_predictions(image, W1, b1, W2, b2)
    print(prediction)


