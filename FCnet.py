import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import readdata
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='FC MNIST classifier')
    parser.add_argument('--work-dir', default='best_model.npz', help='the dir to save logs and models')
    parser.add_argument('--resume-from', default='best_model.npz', help='the checkpoint file to resume from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', default=1, help='learning rate')
    parser.add_argument('--alpha', default=1e-6, help='regularization')
    parser.add_argument('--num_iter', default=2000, help='number of iterations of gradient descent')
    parser.add_argument('--n_H', default=256, help='number of neurons in the hidden layer')
    parser.add_argument('--n', default=784, help='number of pixels in an image')
    parser.add_argument('--K', default=10, help='number of class')
    parser.add_argument('--eval', action='store_true', help='whether to evaluate the checkpoint')
    args = parser.parse_args()
    return args


# relu activation function
# THE fastest vectorized implementation for ReLU
def relu(x):
    x[x < 0] = 0
    return x


# Fully-connected network
def FCnet(X, W, b):
    """
    simple FNN with 1 hidden layer
    Layer 1: input
    Layer 2: hidden layer, with a size implied by the arguments W[0], b
    Layer 3: output layer, with a size implied by the arguments W[1]
    """
    # layer 1 = input layer
    a1 = X
    # layer 1 (input layer) -> layer 2 (hidden layer)
    z1 = np.matmul(a1, W[0]) + b[0]

    # layer 2 activation
    a2 = relu(z1)
    # layer 2 (hidden layer) -> layer 3 (output layer)
    z2 = np.matmul(a2, W[1]) / a2.shape[-1]
    s = np.exp(z2)
    total = np.sum(s, axis=1).reshape(-1, 1)
    sigma = s / total
    # the output is a probability for each sample
    return sigma


def loss(y_pred, y_true):
    """
    Loss function: cross entropy with an L^2 regularization
    y_true: ground truth, of shape (N, )
    y_pred: prediction made by the model, of shape (N, K)
    N: number of samples in the batch
    K: global variable, number of classes
    """
    K = 10  # class number
    N = len(y_true)
    # loss_sample stores the cross entropy for each sample in X
    # convert y_true from labels to one-hot-vector encoding
    y_true_one_hot_vec = (y_true[:, np.newaxis] == np.arange(K))
    loss_sample = (np.log(y_pred) * y_true_one_hot_vec).sum(axis=1)
    # loss_sample is a dimension (N,) array
    # for the final loss, we need take the average
    return -np.mean(loss_sample)


def backprop(W, b, X, y, alpha=1e-4):
    """
    Step 1: explicit forward pass FCnet(X;W,b)
    Step 2: backpropagation for dW and db
    """
    K = 10
    N = X.shape[0]

    # Step 1:
    # layer 1 = input layer
    a1 = X
    # layer 1 (input layer) -> layer 2 (hidden layer)
    z1 = np.matmul(a1, W[0]) + b[0]
    # layer 2 activation
    a2 = relu(z1)

    # one more layer

    # layer 2 (hidden layer) -> layer 3 (output layer)
    z2 = np.matmul(a2, W[1]) / a2.shape[-1]

    s = np.exp(z2)
    total = np.sum(s, axis=1).reshape(-1, 1)
    sigma = s / total

    # Step 2:

    # layer 2->layer 3 weights' derivative
    # delta2 is \partial L/partial z2, of shape (N,K)
    y_one_hot_vec = (y[:, np.newaxis] == np.arange(K))
    delta2 = (sigma - y_one_hot_vec) / a2.shape[-1]
    grad_W1 = np.matmul(a2.T, delta2)

    # layer 1->layer 2 weights' derivative
    # delta1 is \partial a2/partial z1
    # layer 2 activation's (weak) derivative is 1*(z1>0)
    delta1 = np.matmul(delta2, W[1].T) * (z1 > 0)
    grad_W0 = np.matmul(X.T, delta1)

    # Student project: extra layer of derivative

    # no derivative for layer 1

    # the alpha part is the derivative for the regularization
    # regularization = 0.5*alpha*(np.sum(W[1]**2) + np.sum(W[0]**2))

    dW = [grad_W0 / N + alpha * W[0], grad_W1 / N + alpha * W[1]]
    db = [np.mean(delta1, axis=0)]
    # dW[0] is W[0]'s derivative, and dW[1] is W[1]'s derivative; similar for db
    return dW, db


def load(filename='bestmodel.npz'):
    """Prepare a neural network from a compressed binary containing weights
    and biases arrays. Size of layers are derived from dimensions of
    numpy arrays.

    Parameters
    ----------
    filename : str, optional
        Name of the ``.npz`` compressed binary in models directory.

    """
    npz_members = np.load(os.path.join(os.curdir, filename))

    w = list(npz_members['weights'])
    b = list(npz_members['biases'])
    return w, b


def save(w, b, filename='bestmodel.npz'):
    """Save weights, biases and hyperparameters of neural network to a
    compressed binary. This ``.npz`` binary is saved in 'models' directory.

    Parameters
    ----------
    filename : str, optional
        Name of the ``.npz`` compressed binary in to be saved.

    """
    np.savez_compressed(
        file=os.path.join(os.curdir, filename),
        weights=w,
        biases=b,
    )


def main():
    args = parse_args()
    X_train, X_val, Y_train, Y_val, X_test, Y_test = readdata.read_input()
    if args.eval:
        W, b = load(args.resume_from)
        y_pred_final = FCnet(X_test, W, b)
        print("test loss is {:.8}".format(loss(y_pred_final, Y_test)))
        print("test accuracy is {:.4%}".format(np.mean(np.argmax(y_pred_final, axis=1) == Y_test)))
    else:
        # initialization
        np.random.seed(args.seed)
        W = [1e-1 * np.random.randn(args.n, args.n_H), 1e-1 * np.random.randn(args.n_H, args.K)]
        b = [np.random.randn(args.n_H)]
        loss_train_list = []
        loss_val_list = []
        acc_val_list = []
        best_acc = 0
        for i in range(args.num_iter):
            dW, db = backprop(W, b, X_train, Y_train, args.alpha)
            W[0] -= args.lr * dW[0]
            W[1] -= args.lr * dW[1]
            b[0] -= args.lr * db[0]

            # train loss
            y_train_pred = FCnet(X_train, W, b)
            train_loss = loss(y_train_pred, Y_train)
            loss_train_list.append(train_loss)
            # val loss
            y_val_pred = FCnet(X_val, W, b)
            val_loss = loss(y_val_pred, Y_val)
            loss_val_list.append(val_loss)
            # val acc
            acc_val = np.mean(np.argmax(y_val_pred, axis=1) == Y_val)
            acc_val_list.append(acc_val)
            if acc_val > best_acc:
                best_acc = acc_val
                save(W, b, args.work_dir)
            # log
            if i % 100 == 0:
                print("Train loss after", i + 1, "iterations is {:.8}".format(
                    train_loss))
                print("Val loss after", i + 1, "iterations is {:.8}".format(
                    val_loss))
                print("Val accuracy after", i + 1, "iterations is {:.4%}".format(
                    acc_val))
                print("Best val accuracy is {:.4%}".format(best_acc))
        x_plot = range(len(loss_train_list))
        plt.plot(x_plot, loss_train_list, label='train_loss')
        plt.plot(x_plot, loss_val_list, label='val_loss')
        plt.legend()
        plt.title('loss')
        plt.savefig('loss')
        plt.clf()
        plt.plot(x_plot, acc_val_list, label='val_acc')
        plt.legend()
        plt.title('val acc')
        plt.savefig('val_acc')
        y_pred_final = FCnet(X_test, W, b)
        print("Final test loss is {:.8}".format(loss(y_pred_final, Y_test)))
        print("Final test accuracy is {:.4%}".format(np.mean(np.argmax(y_pred_final, axis=1) == Y_test)))


if __name__ == '__main__':
    main()
