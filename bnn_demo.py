from __future__ import division

import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import ml.trainers as trainers
import ml.neural_nets as nn
import ml.loss_functions as lf

wdecay = 0.0001


def create_dataset():
    """
    Creates a small dataset of 2d points in two linearly separable classes.
    :return: datapoints, labels
    """

    data_per_class = 12

    rng_state = rng.get_state()
    rng.seed(0)

    x1 = rng.multivariate_normal([-6, 0], np.eye(2), data_per_class)
    x2 = rng.multivariate_normal([+6, 0], np.eye(2), data_per_class)

    y1 = np.zeros(data_per_class)
    y2 = np.ones(data_per_class)

    xs = np.concatenate([x1, x2], axis=0)
    ys = np.concatenate([y1, y2], axis=0)

    rng.set_state(rng_state)

    return xs, ys


def create_net(svi=False):
    """
    Creates a feedforward neural net.
    :param svi: whether the neural net should be SVI enabled
    :return: the net
    """

    if svi:
        net = nn.FeedforwardNet_SVI(2)
    else:
        net = nn.FeedforwardNet(2)

    net.addLayer(10, 'relu')
    net.addLayer(1, 'logistic')

    return net


def create_grid(xmin, xmax, N):
    """
    Creates a grid for 3d plotting.
    :param xmin: lower limit
    :param xmax: upper limit
    :param N: number of points in the grid per dimension
    :return: the grid
    """

    xx = np.linspace(xmin, xmax, N)
    X, Y = np.meshgrid(xx, xx)
    data = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)

    return data, X, Y


def show_train_data():
    """
    Plots the training data.
    """

    xs, ys = create_dataset()

    plt.figure()
    plt.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)
    plt.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    plt.axis([-12, 12, -12, 12])
    plt.title('Training data')

    plt.show()


def fit_neural_net_demo():
    """
    Fits a non-bayesian neural net to the training data by minimizing cross entropy.
    """

    xs, ys = create_dataset()
    net = create_net()

    # train the net
    trn_target, trn_loss = lf.CrossEntropy(net.output)
    regularizer = lf.WeightDecay(net.parms, wdecay)
    trainer = trainers.SGD(
        model=net,
        trn_data=[xs, ys],
        trn_loss=trn_loss + regularizer / xs.shape[0],
        trn_target=trn_target
    )
    trainer.train(tol=1.0e-9, monitor_every=10, show_progress=True)

    # make predictions
    tst_data, X, Y = create_grid(-12, 12, 50)
    pred = net.eval(tst_data)

    # plot the prediction surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = pred.reshape(list(X.shape))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    ax.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)
    ax.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)
    ax.view_init(elev=90, azim=-90)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    ax.axis([-12, 12, -12, 12])
    fig.suptitle('Prediction surface of trained net')

    plt.show()


def bayesian_neural_net_svi_demo():
    """
    Trains a bayesian neural net on the training set using Stochastic Variational Inference.
    """

    xs, ys = create_dataset()
    net = create_net(svi=True)
    tst_data, X, Y = create_grid(-12, 12, 50)

    # train the net
    trn_target, trn_loss = lf.CrossEntropy(net.output)
    regularizer = lf.SviRegularizer(net.mps, net.sps, wdecay)
    trainer = trainers.SGD(
        model=net,
        trn_data=[xs, ys],
        trn_loss=trn_loss + regularizer / xs.shape[0],
        trn_target=trn_target
    )
    trainer.train(maxepochs=80000, monitor_every=10, show_progress=True)

    # make predictions with zero noise
    base_pred = net.eval(tst_data, rand=False)

    # make predictions by averaging samples
    n_samples = 1000
    avg_pred = 0.0
    for _ in xrange(n_samples):
        avg_pred += net.eval(tst_data, rand=True)
    avg_pred /= n_samples

    # plot the base prediction surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = base_pred.reshape(list(X.shape))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    ax.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)
    ax.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)
    ax.view_init(elev=90, azim=-90)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    ax.axis([-12, 12, -12, 12])
    fig.suptitle('Prediction surface using average weights')

    # plot the average prediction surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = avg_pred.reshape(list(X.shape))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    ax.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)
    ax.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)
    ax.view_init(elev=90, azim=-90)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    ax.axis([-12, 12, -12, 12])
    fig.suptitle('Bayesian prediction surface')

    # plot the sample prediction surfaces
    fig = plt.figure()
    fig.suptitle('Sample prediction surfaces')

    for i in xrange(6):

        sample_pred = net.eval(tst_data, rand=True)

        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        Z = sample_pred.reshape(list(X.shape))
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        ax.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)
        ax.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)
        ax.view_init(elev=90, azim=-90)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.axis('equal')
        ax.axis([-12, 12, -12, 12])

    plt.show()


def bayesian_neural_net_hmc_demo():
    """
    Trains a bayesian neural net on the training set using Hamiltonian Monte Carlo.
    """

    xs, ys = create_dataset()
    net = create_net()
    tst_data, X, Y = create_grid(-12, 12, 50)

    # make predictions on a grid of points
    trn_target, trn_loss = lf.CrossEntropy(net.output)
    regularizer = lf.WeightDecay(net.parms, wdecay)
    sampler = trainers.HMC(
        model=net,
        trn_data=[xs, ys],
        trn_loss=xs.shape[0] * trn_loss + regularizer,
        trn_target=trn_target
    )
    ensemble = sampler.gen(
        n_samples=2000,
        L=100,
        me=0.3,
        show_traces=True
    )
    avg_pred = ensemble.eval(tst_data)

    # plot the prediction surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = avg_pred.reshape(list(X.shape))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
    ax.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)
    ax.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)
    ax.view_init(elev=90, azim=-90)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.axis('equal')
    ax.axis([-12, 12, -12, 12])
    fig.suptitle('Bayesian prediction surface')

    # plot the prediction surfaces of a few sample networks
    fig = plt.figure()
    fig.suptitle('Sample prediction surfaces')

    for c, i in enumerate(rng.randint(0, ensemble.n_diff_models, 6)):

        ax = fig.add_subplot(2, 3, c+1, projection='3d')
        Z = ensemble.eval_model(i, tst_data).reshape(list(X.shape))
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0)
        ax.plot(xs[ys == 0, 0], xs[ys == 0, 1], 'b.', ms=12)
        ax.plot(xs[ys == 1, 0], xs[ys == 1, 1], 'r.', ms=12)
        ax.view_init(elev=90, azim=-90)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.axis('equal')
        ax.axis([-12, 12, -12, 12])

    plt.show()
