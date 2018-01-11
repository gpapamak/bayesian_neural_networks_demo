from itertools import izip
import numpy as np
import theano
import theano.tensor as tt


def select_theano_act_function(name, dtype=theano.config.floatX):
    """
    Given the name of an activation function, returns a handle for the corresponding function in theano.
    """

    if name == 'logistic':
        clip = 15.0 if dtype == 'float32' else 19.0
        f = lambda x: tt.nnet.sigmoid(tt.clip(x, -clip, clip))

    elif name == 'tanh':
        clip = 9.0 if dtype == 'float32' else 19.0
        f = lambda x: tt.tanh(tt.clip(x, -clip, clip))

    elif name == 'linear':
        f = lambda x: x

    elif name == 'relu':
        f = tt.nnet.relu

    elif name == 'softplus':
        f = tt.nnet.softplus

    elif name == 'softmax':
        f = tt.nnet.softmax

    else:
        raise ValueError(name + ' is not a supported activation function type.')

    return f


def copy_model_parms(source_model, target_model):
    """
    Copies the parameters of source_model to target_model.
    """

    for sp, tp in izip(source_model.parms, target_model.parms):
        tp.set_value(sp.get_value())


def one_hot_encode(labels, n_labels):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels])
    y[xrange(labels.size), labels] = 1

    return y


def prepare_cond_input(xy, dtype):
    """
    Prepares the conditional input for model evaluation.
    :param xy: tuple (x, y) for evaluating p(y|x)
    :param dtype: data type
    :return: prepared x, y and flag whether single datapoint input
    """

    x, y = xy
    x = np.asarray(x, dtype=dtype)
    y = np.asarray(y, dtype=dtype)

    one_datapoint = False

    if x.ndim == 1:

        if y.ndim == 1:
            x = x[np.newaxis, :]
            y = y[np.newaxis, :]
            one_datapoint = True

        else:
            x = np.tile(x, [y.shape[0], 1])

    else:

        if y.ndim == 1:
            y = np.tile(y, [x.shape[0], 1])

        else:
            assert x.shape[0] == y.shape[0], 'wrong sizes'

    return x, y, one_datapoint


def are_parms_finite(model):
    """
    Check whether all parameters of a model are finite.
    :param model: an ml model
    :return: False if at least one parameter is inf or nan
    """

    check = True

    for p in model.parms:
        check = check and np.all(np.isfinite(p.get_value()))

    return check
