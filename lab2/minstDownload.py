import tensorflow as tf
import numpy as np


def load_mnist_from_tf():
    # 从 Keras 自带的 mnist 模块下载
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train.shape -> (60000, 28, 28)
    # x_test.shape  -> (10000, 28, 28)

    # 将像素值从 [0, 255] 缩放到 [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 将每张 28x28 的图展开为 784(28*28) 维向量
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_test = x_test.reshape((x_test.shape[0], 28 * 28))

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_mnist_from_tf()
    print("训练集 X_train shape:", X_train.shape)  # (60000, 784)
    print("训练集 y_train shape:", y_train.shape)  # (60000,)
    print("测试集 X_test shape:", X_test.shape)  # (10000, 784)
    print("测试集 y_test shape:", y_test.shape)  # (10000,)