from matplotlib import gridspec
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from scipy import signal

import cupy as cp
import numpy as np

sns.set()

EPS = 1e-16


def relu(x):
    s = cp.maximum(0, x)
    return s


def softmax(x):
    a = cp.exp(x)
    return a / cp.sum(a, axis=0)


def batch_norm(S, mean, var):
    S_hat = cp.dot((cp.linalg.inv(cp.sqrt(cp.diag(var + EPS)))), S - mean)
    return S_hat


class KlayerNN:

    def __init__(self):
        # Data parameters
        self.d = None  # Features
        self.K = None  # Classes

        # Gradient Descent hyperparameters
        self.lambda_L2 = None
        self.batch_size = None
        self.n_epochs = None
        self.eta_min = None
        self.eta_max = None
        self.eta = None

        self.scheduled_eta = None

        self.layers = None
        # Model parameters
        self.W = None
        self.b = None
        self.gamma = None
        self.beta = None

        # Temporary values
        self.all_X = None
        self.all_S_hat = None
        self.all_S = None
        self.all_mean = None
        self.all_var = None

        # History of training
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def compile(self, d, K, layers):
        self.d = d
        self.K = K.item()
        self.layers = layers
        self.W = []
        self.b = []
        self.gamma = []
        self.beta = []

        # Input layer
        m = layers[0]
        self.W.append(
            cp.random.normal(loc=0.0, scale=1 / cp.sqrt(d), size=(m, d))
        )
        self.b.append(cp.zeros((m, 1)))
        self.gamma.append(1)
        self.beta.append(0)
        # k-2 hidden layers
        for i, m in enumerate(layers[1:]):
            self.W.append(
                cp.random.normal(loc=0.0, scale=1 / cp.sqrt(layers[i]), size=(m, layers[i]))
            )
            self.b.append(cp.zeros((m, 1)))
            self.gamma.append(1)
            self.beta.append(0)

        # Output layer
        self.W.append(
            cp.random.normal(loc=0.0, scale=1 / cp.sqrt(layers[-1]), size=(self.K, layers[-1]))
        )
        self.b.append(cp.zeros((self.K, 1)))

    def _saveGDparams(self, GDparams):
        self.lambda_L2 = GDparams['lambda_L2']
        self.batch_size = GDparams['batch_size']
        self.n_epochs = GDparams['n_epochs']
        self.n_s = GDparams['n_s']
        self.eta_min = GDparams['eta_min']
        self.eta_max = GDparams['eta_max']

        # Prepare schedule of the learning rate
        t = np.arange(self.n_s * 2)
        freq = 1 / (2 * self.n_s)
        self.scheduled_eta = (signal.sawtooth(2 * np.pi * t * freq, 0.5) + 1) / 2 * (
                self.eta_max - self.eta_min) + self.eta_min

        # Pre compute or allocate for saving computations
        self.ones_nb = cp.ones((self.batch_size, 1))
        self.c = 1 / self.batch_size

        # Debug
        # plt.plot(self.scheduled_eta)
        # plt.show()

    def forward_pass(self, X):
        """
        Run a forward pass storing the S(l) inside the Python class.

        Parameters:
            X: Input data

        Returns:
            Softmax probabilities.
        """
        self.all_X = [X]
        self.all_S_hat = []
        self.all_S = []
        self.all_mean = []
        self.all_var = []
        for k, (W, b, gamma, beta) in enumerate(zip(self.W[:-1], self.b[:-1], self.gamma, self.beta)):
            S = self.all_X[k]
            S = W.dot(S) + b
            self.all_S.append(S)
            mean, var = S.mean(axis=1, keepdims=True), S.var(axis=1)
            self.all_mean.append(mean)
            self.all_var.append(var)
            S = batch_norm(S, mean, var)
            self.all_S_hat.append(S)
            S = gamma * S + beta
            S = relu(S)
            self.all_X.append(S)

        S = self.W[-1].dot(self.all_X[-1]) + self.b[-1]
        self.all_S.append(S)
        P = softmax(S)
        return P

    def fast_forward_pass(self, X):
        """
        Runs a faster forward pass without storing all the S(l).
        This should be used for evaluations at the end of each epoch.

        Parameters:
            X: Input data.
        Returns:
            A tuple with (last layer logits, softmax probabilities).
        """
        S = X
        for W, b, gamma, beta in zip(self.W[:-1], self.b[:-1], self.gamma, self.beta):
            S = W.dot(S) + b
            mean, var = S.mean(axis=1, keepdims=True), S.var(axis=1)
            S = batch_norm(S, mean, var)
            S = gamma * S + beta
            S = relu(S)

        S = self.W[-1].dot(S) + self.b[-1]
        P = softmax(S)
        return S, P

    def compute_cost(self, X, y, P=None, fast=False):
        N = X.shape[1]
        if P is None:
            if fast:
                P = self.fast_forward_pass(X)[1]
            else:
                P = self.forward_pass(X)

        loss = -cp.log(P[y, cp.arange(N)]).mean()
        reg_term = self.lambda_L2 * np.sum([cp.square(W).sum() for W in self.W])

        return (loss + reg_term).item(), P

    def compute_accuracy(self, X, y, P=None, fast=False):
        if P is None:
            if fast:
                P = self.fast_forward_pass(X)[1]
            else:
                P = self.forward_pass(X)
        predictions = cp.argmax(P, axis=0)
        accuracy = cp.mean(predictions == y)
        return accuracy.item()

    def evaluate(self, X, y, fast=True):
        """
        Computes cost and accuracy of the given data.
        """
        cost, P = self.compute_cost(X, y, fast=fast)
        accuracy = self.compute_accuracy(X, y, P, fast=fast)
        return cost, accuracy

    def backward_pass(self, X, Y, P):
        c = self.c
        k = len(self.layers) + 1  # hidden + output
        ones_nb = self.ones_nb
        X_all = self.all_X

        # (21)
        G = P - Y
        # (22)
        dJ_dW_k = c * G.dot(X_all[k - 1].T) + 2 * self.lambda_L2 * self.W[k - 1]
        dJ_db_k = c * G.dot(ones_nb)
        # (23)
        G = self.W[k - 1].T.dot(G)
        # (24)
        G = G * (X_all[k - 1] > 0)

        dJ_dgamma_l = []
        dJ_dbeta_l = []
        dJ_dW_l = []
        dJ_db_l = []
        for l in range(k - 2, -1, -1):
            # (25)
            dJ_dgamma_l.append(c * (G * self.all_S_hat[l]).dot(ones_nb))
            dJ_dbeta_l.append(c * G.dot(ones_nb))
            # (26)
            G = G * self.gamma[l] * ones_nb.T
            # (27)
            G = self.BN_backpass(G, self.all_S_hat[l], self.all_mean[l], self.all_var[l])
            # (28)
            dJ_dW_l.append(c * G.dot(self.all_X[l].T) + 2 * self.lambda_L2 * self.W[l])
            dJ_db_l.append(c * G.dot(ones_nb))

            if l >= 1:
                # (29)
                G = self.W[l].T.dot(G)
                # (30)
                G = G * (self.all_X[l] > 0)

        dJ_dW = list(reversed(dJ_dW_l)) + [dJ_dW_k]
        dJ_db = list(reversed(dJ_db_l)) + [dJ_db_k]
        dJ_dgamma = list(reversed(dJ_dgamma_l))
        dJ_dbeta = list(reversed(dJ_dbeta_l))

        return dJ_dW, dJ_db, dJ_dgamma, dJ_dbeta

    def BN_backpass(self, G, S, mu, v):
        oneT = self.ones_nb.T
        sigma1 = ((v + EPS) ** -0.5).reshape(-1, 1)
        sigma2 = ((v + EPS) ** -1.5).reshape(-1, 1)
        G1 = G * (sigma1.dot(oneT))
        G2 = G * (sigma2.dot(oneT))
        D = S - mu.dot(oneT)
        c = (G2 * D).dot(oneT.T)
        G = G1 - self.c * (G1.dot(oneT.T)).dot(oneT) - self.c * D * (c.dot(oneT))  # TODO: can be optimized
        return G

    def fit(self, train_data, GDparams, val_data=None, val_split=None):
        assert val_data is not None or val_split is not None, 'Validation set not defined.'
        self._saveGDparams(GDparams)
        if val_data is None:
            train_data, val_data = self.split_data(train_data, val_split)

        N = train_data[0].shape[1]

        self._run_epochs(train_data, val_data, N)

        return self.history

    def _run_epochs(self, train_data, val_data, N, shuffle=True):
        n_batches = N // self.batch_size

        for_epoch = trange(self.n_epochs, leave=True, unit='epoch')

        for epoch in for_epoch:
            if shuffle:
                self.shuffleData(train_data)

            self._run_batches(train_data, n_batches, epoch)

            # Evaluate for saving in history
            train_loss, train_acc, val_loss, val_acc = self._update_history(train_data, val_data)

            for_epoch.set_description(f'train_loss: {train_loss:.4f}\ttrain_acc: {100 * train_acc:.2f}%' + ' | ' +
                                      f'val_loss: {val_loss:.4f}\ttrain_acc: {100 * val_acc:.2f}% ')


    def _run_batches(self, train_data, n_batches, epoch):

        for b in range(n_batches):
            X_batch, Y_batch = self._get_mini_batch(train_data, b)

            self._update_eta(b, n_batches, epoch)

            self._update_weights(X_batch, Y_batch)

    def _update_weights(self, X_batch, Y_batch):
        P = self.forward_pass(X_batch)
        dJ_dW, dJ_db, dJ_dgamma, dJ_dbeta = self.backward_pass(X_batch, Y_batch, P)

        self.W[0] -= self.eta * dJ_dW[0] + 2 * self.lambda_L2 * self.W[0]
        self.b[0] -= self.eta * dJ_db[0]

        for i, (dW, db, dgamma, dbeta) in enumerate(zip(dJ_dW[1:], dJ_db[1:], dJ_dgamma, dJ_dbeta)):
            self.W[i + 1] -= self.eta * dW + 2 * self.lambda_L2 * self.W[i + 1]
            self.b[i + 1] -= self.eta * db
            self.gamma[i] -= self.eta * dgamma
            self.beta[i] -= self.eta * dbeta

    def _get_mini_batch(self, data, j):
        j_start = j * self.batch_size
        j_end = (j + 1) * self.batch_size
        X_batch = data[0][:, j_start:j_end]
        Y_batch = data[1][:, j_start:j_end]
        return X_batch, Y_batch

    def shuffleData(self, data):
        """
        Shuffle a dataset made of X, Y and y.
        """
        N = data[0].shape[1]
        indeces = np.arange(N)
        np.random.shuffle(indeces)
        data[0][:] = data[0][:, indeces]
        data[1][:] = data[1][:, indeces]
        data[2][:] = data[2][indeces]

    def _update_history(self, train_data, val_data):
        train_loss, train_acc = self.evaluate(train_data[0], train_data[2], fast=True)
        val_loss, val_acc = self.evaluate(val_data[0], val_data[2], fast=True)

        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

        return train_loss, train_acc, val_loss, val_acc

    def _update_eta(self, b, batches, epoch):

        t = batches * epoch + b
        index = t % (self.n_s * 2)
        self.eta = self.scheduled_eta[index]

    def split_data(self, train_data, val_split):
        self.shuffleData(train_data)
        train_data_new = []
        val_data = []

        N = train_data[0].shape[1]
        end_train = N - int(N * val_split)

        train_data_new.append(train_data[0][:, :end_train])
        train_data_new.append(train_data[1][:, :end_train])
        train_data_new.append(train_data[2][:end_train])

        val_data.append(train_data[0][:, end_train:])
        val_data.append(train_data[1][:, end_train:])
        val_data.append(train_data[2][end_train:])

        return train_data_new, val_data


def load_dataset():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = cp.asarray(X_train).reshape(X_train.shape[0], -1).T
    X_test = cp.asarray(X_test).reshape(X_test.shape[0], -1).T
    y_train = cp.asarray(y_train).reshape(-1)
    y_test = cp.asarray(y_test).reshape(-1)
    Y_train = make_one_hot(y_train).T
    Y_test = make_one_hot(y_test).T
    return (X_train, y_train, Y_train), (X_test, y_test, Y_test)


def make_one_hot(x):
    one_hot_x = cp.zeros((x.size, x.max().item() + 1))
    one_hot_x[cp.arange(x.size), x] = 1
    return one_hot_x


def plot_history(history, GDparams):
    fig = plt.figure(1, figsize=(12, 5))
    gs1 = gridspec.GridSpec(1, 2)
    ax = [fig.add_subplot(subplot) for subplot in gs1]

    ax[0].set_title('Cost')
    ax[0].plot(history['train_loss'], sns.xkcd_rgb["pale red"], label='Train')
    ax[0].plot(history['val_loss'], sns.xkcd_rgb["denim blue"], label='Val')
    ax[0].axhline(history['test_loss'], color=sns.xkcd_rgb["medium green"], label='Test')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()

    ax[1].set_title('Accuracy')
    ax[1].plot(history['train_acc'], sns.xkcd_rgb["pale red"], label='Train')
    ax[1].plot(history['val_acc'], sns.xkcd_rgb["denim blue"], label='Val')
    ax[1].axhline(history['test_acc'], color=sns.xkcd_rgb["medium green"], label='Test')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()

    main_title = 'batch_size={0}, $\eta_{{min}}={1}$, $\eta_{{max}}={2}$, $n_s={3}$'.format(
        GDparams['batch_size'], GDparams['eta_min'], GDparams['eta_max'], GDparams['n_s']
    ) + ', $\lambda={0}$, n_epochs={1}'.format(GDparams['lambda_L2'], GDparams['n_epochs'])
    fig.suptitle(main_title, fontsize=18)

    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

    fig.show()


if __name__ == '__main__':
    # Loading dataset
    (X_train, y_train, Y_train), (X_test, y_test, Y_test) = load_dataset()
    # Data characteristics
    d, N = X_train.shape
    K = y_train.max() + 1

    # Gradient Descent parameters
    GDparams = {'batch_size': 128,
                'eta_min': 1e-5,
                'eta_max': 1e-1,
                'n_s': 800,
                'n_epochs': 100,
                'lambda_L2': 0.0}

    model = KlayerNN()
    model.compile(d, K, layers=[50, 30])

    history = model.fit((X_train, Y_train, y_train), GDparams, val_split=0.1)

    history['test_loss'], history['test_acc'] = model.evaluate(X_test, y_test)
    print('Test loss: {0:.4f}\tTest acc: {1:.2f}%'.format(history['test_loss'], history['test_acc'] * 100))
    plot_history(history, GDparams)
