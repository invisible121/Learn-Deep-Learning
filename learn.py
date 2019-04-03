import skimage

from lr_utils import load_dataset
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

"""
w.shape = (dim, 1)
X.shape = (px * px * 3, num_of_pic)

Y.shape = (1, ...)
"""


# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 初始化
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    return w, b


# 前向传播
def propagate(w, b, X, Y):
    A = sigmoid(np.dot(w.T, X) + b)

    m = X.shape[1]

    cost = np.sum(np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T)) / (-m)

    dw = np.dot(X, (A - Y).T) / m

    db = np.sum(A - Y) / m

    return cost, dw, db


# 梯度下降
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):

        cost, dw, db = propagate(w, b, X, Y)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if print_cost == True and i % 100 == 0:
            costs.append(cost)

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return params, grads, costs


def predict(w, b, X):

    Y_possibility = sigmoid(np.dot(w.T, X) + b)

    Y_predict = np.zeros((1, X.shape[1]))

    for i in range(X.shape[1]):
        if Y_possibility[0, i] > 0.5:
            Y_predict[0, i] = 1
        else:
            Y_predict[0, i] = 0

    return Y_predict, Y_possibility


def training(X_train, Y_train, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    return params, grads, costs


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)


def loadAndformatSet():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T

    return train_set_x, classes


if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255
    test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255
    num_px = train_set_x_orig.shape[1]
    params, grads, costs = training(train_set_x, train_set_y, num_iterations=2000, learning_rate=0.009, print_cost=False)

    index = 1

    plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))

    plt.show()
    # my_image = "my_image.jpg"
    while True:
        print("please enter the picture:  ")
        my_image = input()
        fname = "images/" + my_image
        image = np.array(plt.imread(fname))
    #     # , flatten = False
    #
        my_image = skimage.transform.resize(image, output_shape=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
        # , size = (num_px, num_px)).reshape((1, num_px * num_px * 3)
        my_predicted_image, my_predicted_possibility = predict(params["w"], params["b"], my_image)
        print("y = " + str(np.squeeze(my_predicted_image)))
        print(my_predicted_image.squeeze())
        print("is a " + str(classes[int(np.squeeze(my_predicted_image))]))
        print("possibility : " + str(my_predicted_possibility))
        plt.imshow(image)
        im = Image.open(fname)
        # im.show()
        # im = plt.imread(fname)
        # plt.imshow(im)
    #
    #     print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
    #         int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
    #
        plt.show()
