import numpy as np
def save_model(W1,b1,W2,b2):

    np.savez("params.npz", W1=W1, b1=b1, W2=W2, b2=b2)

def load_model():
    npz = np.load("params.npz")
    W1 = npz["W1"]
    b1 = npz["b1"]
    W2 = npz["W2"]
    b2 = npz["b2"]
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def ReLU_deriv(Z):
    return Z > 0
