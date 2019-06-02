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


cp.random.seed(100)
np.random.seed(100)


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
        self.BN = None
        self.lambda_L2 = None
        self.batch_size = None
        self.n_epochs = None
        self.n_cycles = None
        self.eta_min = None
        self.eta_max = None
        self.eta = None
        self.alpha_exp_avg = None
        self.sigma_init = None

        self.layers = None

        self.scheduled_eta = None

        # Flag for checking the gradient numerically
        self.check_gradient = False

        self.first_bath = True

        # Model parameters
        self.W = None
        self.b = None
        self.gamma = None
        self.beta = None

        self.compensation = None

        # Temporary values
        self.all_X = None
        self.all_S_hat = None
        self.all_S = None
        self.all_mean = None
        self.all_var = None

        # History of training
        self.history = {'train_cost': [], 'train_loss': [], 'train_acc': [],
                        'val_cost': [], 'val_loss': [], 'val_acc': [], 'eta': []}

    def compile(self, d, K, layers):
        self.d = d
        self.K = K.item()
        self.layers = layers
        self.W = []
        self.b = []
        self.gamma = []
        self.beta = []

        self._init_weights()

    def _init_weights(self):
        layers = self.layers
        d = self.d
        K = self.K
        # Input layer
        m = layers[0]
        std = cp.sqrt(2 / d) if self.sigma_init is None else self.sigma_init
        self.W.append(
            cp.random.normal(loc=0.0, scale=std, size=(m, d))
        )
        self.b.append(cp.zeros((m, 1)))
        self.gamma.append(cp.ones((m, 1)))
        self.beta.append(cp.zeros((m, 1)))
        # k-2 hidden layers
        for i, m in enumerate(layers[1:]):
            std = cp.sqrt(2 / layers[i]) if self.sigma_init is None else self.sigma_init
            self.W.append(
                cp.random.normal(loc=0.0, scale=std, size=(m, layers[i]))
            )
            self.b.append(cp.zeros((m, 1)))
            self.gamma.append(cp.ones((m, 1)))
            self.beta.append(cp.zeros((m, 1)))

        # Output layer
        std = cp.sqrt(2 / layers[-1]) if self.sigma_init is None else self.sigma_init
        self.W.append(
            cp.random.normal(loc=0.0, scale=std, size=(K, layers[-1]))
        )
        self.b.append(cp.zeros((K, 1)))

        self.all_mean = [0] * len(self.beta)
        self.all_var = [0] * len(self.beta)

    def _saveGDparams(self, GDparams):
        self.lambda_L2 = GDparams['lambda_L2']
        self.batch_size = GDparams['batch_size']
        self.n_s = GDparams['n_s']
        self.eta_min = GDparams['eta_min']
        self.eta_max = GDparams['eta_max']
        self.alpha_exp_avg = GDparams['alpha_exp_avg']
        self.n_cycles = GDparams['n_cycles']
        self.BN = GDparams['BN']
        self.sigma_init = GDparams['sigma_init']

        # Prepare schedule of the learning rate
        t = np.arange(self.n_s * 2)
        freq = 1 / (2 * self.n_s)
        self.scheduled_eta = (signal.sawtooth(2 * np.pi * t * freq, 0.5) + 1) / 2 * (
                self.eta_max - self.eta_min) + self.eta_min

        # Pre compute or allocate for saving computations
        self.ones_nb = cp.ones((self.batch_size, 1))
        self.c = 1 / self.batch_size
        self.compensation = (self.batch_size - 1) / self.batch_size
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
        for k, (W, b, gamma, beta) in enumerate(zip(self.W[:-1], self.b[:-1], self.gamma, self.beta)):
            S = self.all_X[k]
            S = W.dot(S) + b
            self.all_S.append(S)
            if self.BN:
                mean, var = S.mean(axis=1, keepdims=True), S.var(axis=1)
                var *= self.compensation
                if self.first_bath:
                    self.all_mean[k] = mean
                    self.all_var[k] = var
                else:
                    self.all_mean[k] = self.alpha_exp_avg * self.all_mean[k] + (1 - self.alpha_exp_avg) * mean
                    self.all_var[k] = self.alpha_exp_avg * self.all_var[k] + (1 - self.alpha_exp_avg) * var
                S = batch_norm(S, self.all_mean[k], self.all_var[k])
                self.all_S_hat.append(S)
                S = gamma * S + beta
            S = relu(S)
            self.all_X.append(S)

        if self.first_bath:
            self.first_bath = False

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
            if self.BN:
                mean, var = S.mean(axis=1, keepdims=True), S.var(axis=1)
                var *= self.compensation
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

        return (loss + reg_term).item(), loss.item(), P

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
        cost, loss, P = self.compute_cost(X, y, fast=fast)
        accuracy = self.compute_accuracy(X, y, P, fast=fast)
        return cost, loss, accuracy

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
            if self.BN:
                # (25)
                dJ_dgamma_l.append(c * (G * self.all_S_hat[l]).dot(ones_nb))
                dJ_dbeta_l.append(c * G.dot(ones_nb))
                # (26)
                G = G * self.gamma[l].dot(ones_nb.T)
                # (27)
                G = self.BN_backpass(G, self.all_S[l], self.all_mean[l], self.all_var[l])
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

        t_tot = self.n_cycles * 2 * self.n_s
        self.n_epochs = int(float(t_tot) * self.batch_size / N)

        self._run_epochs(train_data, val_data, N)

        return self.history

    def _run_epochs(self, train_data, val_data, N, shuffle=True):
        n_batches = N // self.batch_size

        for_epoch = trange(self.n_epochs, leave=True, unit='epoch')

        # Initial evaluation
        train_cost, train_acc, val_cost, val_acc = self._update_history(train_data, val_data)

        for_epoch.set_description(f'train_loss: {train_cost:.4f}\ttrain_acc: {100 * train_acc:.2f}%' + ' | ' +
                                  f'val_loss: {val_cost:.4f}\tval_acc: {100 * val_acc:.2f}% ')

        for epoch in for_epoch:
            if shuffle:
                self.shuffleData(train_data)

            self._run_batches(train_data, n_batches, epoch)

            # Evaluate for saving in history
            train_cost, train_acc, val_cost, val_acc = self._update_history(train_data, val_data)

            for_epoch.set_description(f'train_loss: {train_cost:.4f}\ttrain_acc: {100 * train_acc:.2f}%' + ' | ' +
                                      f'val_loss: {val_cost:.4f}\tval_acc: {100 * val_acc:.2f}% ')

    def _run_batches(self, train_data, n_batches, epoch):

        for b in range(n_batches):
            # print(f'batch: {b}')
            X_batch, Y_batch, y_batch = self._get_mini_batch(train_data, b)

            self._update_eta(b, n_batches, epoch)

            self._update_weights(X_batch, Y_batch, y_batch)

            # train_cost, train_loss, train_acc = self.evaluate(train_data[0], train_data[2])

            # print(f'train_loss: {train_cost:.4f}\ttrain_acc: {100 * train_acc:.2f}%')

    def _update_weights(self, X_batch, Y_batch, y_batch):
        P = self.forward_pass(X_batch)
        dJ_dW, dJ_db, dJ_dgamma, dJ_dbeta = self.backward_pass(X_batch, Y_batch, P)

        if self.check_gradient:
            self._check_gradient_numerically(X_batch, Y_batch, y_batch, N=10, d=10)

        if self.BN:
            self.W[0] -= self.eta * dJ_dW[0]
            self.b[0] -= self.eta * dJ_db[0]
            for i, (dW, db, dgamma, dbeta) in enumerate(zip(dJ_dW[1:], dJ_db[1:], dJ_dgamma, dJ_dbeta)):
                self.W[i + 1] -= self.eta * dW
                self.b[i + 1] -= self.eta * db
                self.gamma[i] -= self.eta * dgamma
                self.beta[i] -= self.eta * dbeta
        else:
            for i, (dW, db) in enumerate(zip(dJ_dW, dJ_db)):
                self.W[i] -= self.eta * dW
                self.b[i] -= self.eta * db

    def _get_mini_batch(self, data, j):
        j_start = j * self.batch_size
        j_end = (j + 1) * self.batch_size
        X_batch = data[0][:, j_start:j_end]
        Y_batch = data[1][:, j_start:j_end]
        y_batch = data[2][j_start:j_end]
        return X_batch, Y_batch, y_batch

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
        train_cost, train_loss, train_acc = self.evaluate(train_data[0], train_data[2])
        val_cost, val_loss, val_acc = self.evaluate(val_data[0], val_data[2])

        self.history['train_cost'].append(train_cost)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_cost'].append(val_cost)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

        return train_cost, train_acc, val_cost, val_acc

    def _update_eta(self, b, batches, epoch):

        t = batches * epoch + b
        index = t % (self.n_s * 2)
        self.eta = self.scheduled_eta[index]
        self.history['eta'].append(self.eta)

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

    """
    Functions for numerical check of the gradient
    """

    def forward_pass_num(self, X, W1, W2):
        S1 = W1.dot(X) + self.b1
        H = np.maximum(0, S1)
        S = W2.dot(H) + self.b2
        P = softmax(S)
        return H, P

    def compute_cost_num(self, X, y, W1, W2, P=None):
        N = X.shape[1]
        if P is None:
            P = self.forward_pass_num(X, W1, W2)[1]

        loss = -np.log(P[y, np.arange(N)]).mean()
        reg_term = self.lambda_L2 * (np.square(W1).sum() + np.square(W2).sum())

        return loss + reg_term, loss

    def backward_pass_num(self, X, Y, W1, W2, H, P):
        ones_nb = np.ones(X.shape[1])
        c = 1 / X.shape[1]
        G = P - Y
        dL_dW2 = c * G.dot(H.T)
        dL_db2 = c * G.dot(ones_nb).reshape(self.K, 1)
        G = W2.T.dot(G)
        G = G * (H > 0)
        dL_dW1 = c * G.dot(X.T)
        dL_db1 = c * G.dot(ones_nb).reshape(W1.shape[0], 1)

        return dL_dW1, dL_db1, dL_dW2, dL_db2

    def _compute_grad_num_slow(self, X, y, h=1e-5):

        grad_all_b = []
        for j in range(len(self.b)):
            grad_b = cp.zeros(self.b[j].shape)
            for i in range(len(self.b[j])):
                self.b[j][i] -= h
                c1 = self.compute_cost(X, y, fast=True)[0]
                self.b[j][i] += 2 * h
                c2 = self.compute_cost(X, y, fast=True)[0]
                self.b[j][i] -= h
                grad_b[i] = (c2 - c1) / (2 * h)
            grad_all_b.append(grad_b)

        grad_all_W = []
        for j in range(len(self.W)):
            grad_W = cp.zeros(self.W[j].shape)
            for i in range(len(self.W[j])):
                self.W[j][i] -= h
                c1 = self.compute_cost(X, y, fast=True)[0]
                self.W[j][j] += 2 * h
                c2 = self.compute_cost(X, y, fast=True)[0]
                self.W[j][i] -= h
                grad_W[i] = (c2 - c1) / (2 * h)
            grad_all_W.append(grad_W)

        if self.BN:
            grad_all_gamma = []
            for j in range(len(self.gamma)):
                grad_gamma = cp.zeros(self.gamma[j].shape)
                for i in range(len(self.gamma[j])):
                    self.gamma[j][i] -= h
                    c1 = self.compute_cost(X, y, fast=True)[0]
                    self.gamma[j][i] += 2 * h
                    c2 = self.compute_cost(X, y, fast=True)[0]
                    self.gamma[j][i] -= h
                    grad_gamma[i] = (c2 - c1) / (2 * h)
                grad_all_gamma.append(grad_gamma)

            grad_all_beta = []
            for j in range(len(self.beta)):
                grad_beta = cp.zeros(self.beta[j].shape)
                for i in range(len(self.beta[j])):
                    self.beta[j][i] -= h
                    c1 = self.compute_cost(X, y, fast=True)[0]
                    self.beta[j][i] += 2 * h
                    c2 = self.compute_cost(X, y, fast=True)[0]
                    self.beta[j][i] -= h
                    grad_beta[i] = (c2 - c1) / (2 * h)
                grad_all_beta.append(grad_beta)

            return grad_all_W, grad_all_b, grad_all_gamma, grad_all_beta
        return grad_all_W, grad_all_b

    def _check_gradient_numerically(self, X_batch, Y_batch, y_batch, N, d):
        """
        Parameters:
            N: Number of samples to test
            d: Number of features to test
        """

        X = X_batch[:d, :N]
        Y = Y_batch[:, :N]
        y = y_batch[:N]
        backup_W = cp.copy(self.W[0])
        self.W[0] = self.W[0][:, :d]
        self.c = 1 / N
        self.ones_nb = cp.ones((N, 1))

        if self.BN:
            grad_all_W, grad_all_b, grad_all_gamma, grad_all_beta = self._compute_grad_num_slow(X, y)

            P = self.forward_pass(X)
            dJ_dW, dJ_db, dJ_dgamma, dJ_dbeta = self.backward_pass(X, Y, P)

            ok = self._compute_numerical_error(grad_all_W[0], dJ_dW[0])
            ok = self._compute_numerical_error(grad_all_b[0], dJ_db[0])
            for i, (dW_num, dW, db_num, db, dgamma_num, dgamma, dbeta_num, dbeta) in enumerate(
                zip(grad_all_W[1:], dJ_dW[1:], grad_all_b[1:], dJ_db[1:], grad_all_gamma, dJ_dgamma, grad_all_beta, dJ_dbeta)):
                ok = self._compute_numerical_error(dW_num[i+1], dW[i+1])
                ok = self._compute_numerical_error(db_num[i+1], db[i+1])
                ok = self._compute_numerical_error(dgamma_num[i], dgamma[i])
                ok = self._compute_numerical_error(dbeta_num[i], dbeta[i])
        else:
            grad_all_W, grad_all_b = self._compute_grad_num_slow(X, y)

            P = self.forward_pass(X)
            dJ_dW, dJ_db, _, _ = self.backward_pass(X, Y, P)
            for i, (dW_num, dW, db_num, db) in enumerate(
                zip(grad_all_W[1:], dJ_dW[1:], grad_all_b[1:], dJ_db[1:])):
                ok = self._compute_numerical_error(dW_num[i], dW[i])
                ok = self._compute_numerical_error(db_num[i], db[i])

    def _compute_numerical_error(self, A_num, A_check):
        eps = 1e-16
        tolerance_error = 1e-8
        num = np.abs(A_check - A_num)
        den = np.maximum(eps, np.abs(A_num) + np.abs(A_check))
        err = num / den
        max_err = err.max()
        n_ok = (err < tolerance_error).sum()
        p_ok = n_ok / A_num.size * 100

        print(f'Max error: {max_err}\nPercentage of values under max tolerated value: {p_ok}\n' +
              f'eps: {eps}\tMax tolerated error: {tolerance_error}')

        return p_ok


def load_dataset():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = cp.asarray(X_train).reshape(X_train.shape[0], -1).T
    X_test = cp.asarray(X_test).reshape(X_test.shape[0], -1).T
    mean = cp.mean(X_train, axis=1, keepdims=True)
    std = cp.std(X_train, axis=1, keepdims=True)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    y_train = cp.asarray(y_train).reshape(-1)
    y_test = cp.asarray(y_test).reshape(-1)
    Y_train = make_one_hot(y_train).T
    Y_test = make_one_hot(y_test).T
    return (X_train, y_train, Y_train), (X_test, y_test, Y_test)


def make_one_hot(x):
    one_hot_x = cp.zeros((x.size, x.max().item() + 1))
    one_hot_x[cp.arange(x.size), x] = 1
    return one_hot_x


def plot_history(history, GDparams, layers):
    fig = plt.figure(1, figsize=(14, 4.5))
    gs1 = gridspec.GridSpec(1, 3)
    ax = [fig.add_subplot(subplot) for subplot in gs1]

    for i, (title, metric) in enumerate(zip(['Cost', 'Loss', 'Accuracy'], ['cost', 'loss', 'acc'])):
        ax[i].set_title(title)
        ax[i].plot(history[f'train_{metric}'], sns.xkcd_rgb["pale red"], label='Train')
        ax[i].plot(history[f'val_{metric}'], sns.xkcd_rgb["denim blue"], label='Val')
        ax[i].axhline(history[f'test_{metric}'], color=sns.xkcd_rgb["medium green"], label='Test')
        ax[i].text(0.5, history[f'test_{metric}'],
                   '{0:.4f}'.format(history[f'test_{metric}']), fontsize=12, va='center', ha='center',
                   backgroundcolor='w')
        ax[i].set_ylim(bottom=0)
        ax[i].set_xlabel('Epochs')
        ax[i].legend()

    BN_str = 'with' if GDparams['BN'] else 'without'
    main_title = 'batch_size={0}, $\eta_{{min}}={1}$, $\eta_{{max}}={2}$, $n_s={3}$'.format(
        GDparams['batch_size'], GDparams['eta_min'], GDparams['eta_max'], GDparams['n_s']
    ) + ', $\lambda={0}$, n_cycles={1}, {2} BatchNorm'.format(GDparams['lambda_L2'],
                                                              GDparams['n_cycles'],
                                                              BN_str) + '\nhidden layers: {0}'.format(str(layers))
    fig.suptitle(main_title, fontsize=18)

    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.90])

    fig.show()

    fig.savefig('experiment.pdf'.format(GDparams['lambda_L2']))

    # plt.plot(history['eta'])
    # plt.show()


if __name__ == '__main__':
    # Loading dataset
    (X_train, y_train, Y_train), (X_test, y_test, Y_test) = load_dataset()

    # Data characteristics
    d, N = X_train.shape
    K = y_train.max() + 1

    test_cost = []
    test_acc = []
    all_lambda = []
    # for val_lambda in np.arange(0.004, 0.006, 0.0001):
    # Gradient Descent parameters
    # for lambda_L2 in np.arange(0.001, 0.1,):
    batch_size = 100
    GDparams = {'BN': False,
                'batch_size': batch_size,
                'eta_min': 1e-5,
                'eta_max': 1e-1,
                'n_s': 5 * 45000 // batch_size,
                'n_cycles': 2,
                'lambda_L2': 0.0052,
                'alpha_exp_avg': 0.9,
                'sigma_init': 1e-4}
    # layers = [50, 30, 20, 20, 10, 10, 10, 10]
    layers = [50, 50]

    model = KlayerNN()
    model.compile(d, K, layers=layers)

    history = model.fit((X_train, Y_train, y_train), GDparams, val_split=0.1)

    history['test_cost'], history['test_loss'], history['test_acc'] = model.evaluate(X_test, y_test)
    print('Test cost: {0:.4f}\tTest acc: {1:.2f}%'.format(history['test_cost'], history['test_acc'] * 100))

    # test_cost.append(history['test_cost'])
    # test_acc.append(history['test_acc'])
    # print(val_lambda)
    # all_lambda.append(val_lambda)
    # best_i = np.argmax(test_acc)
    # print(f'Best test acc: {test_acc[best_i]}\tLambda: {all_lambda[best_i]}')
    #
    # best_i = np.argmax(test_acc)
    # print(f'Best test acc: {test_acc[best_i]}\tLambda: {all_lambda[best_i]}')
    # plt.plot(all_lambda, test_acc)
    # plt.xlabel('Lambda L2')
    # plt.ylabel('Test accuracy')
    # plt.savefig('lambda_search.pdf')
    # plt.show()
    plot_history(history, GDparams, layers)
