from __future__ import division
from scipy.misc import logsumexp
from copy import deepcopy
from itertools import izip
import numpy as np


class Ensemble:
    """
    Implements an ensemble of other models.
    """

    def __init__(self):
        """Initializes the ensemble as empty."""

        self.models = []
        self.n_copies = []
        self.n_models = 0
        self.n_diff_models = 0
        self.n_inputs = 0
        self.n_outputs = 0

    def add_new(self, model, copy=False):
        """Adds a new model to the ensemble."""

        if self.n_models == 0:
            self.n_inputs = model.n_inputs
            self.n_outputs = model.n_outputs

        else:
            assert self.n_inputs == model.n_inputs
            assert self.n_outputs == model.n_outputs

        if copy:
            self.models.append(deepcopy(model))
        else:
            self.models.append(model)
        self.n_copies.append(1)
        self.n_models += 1
        self.n_diff_models += 1

    def add_existing(self, i):
        """Adds an extra copy of model i in the ensemble."""

        self.n_copies[i] += 1
        self.n_models += 1

    def remove(self, i):
        """Removes a model at position i from the ensemble."""

        self.n_copies[i] -= 1
        if self.n_copies[i] == 0:
            del self.models[i]
            del self.n_copies[i]
            self.n_diff_models -= 1

        self.n_models -= 1
        if self.n_models == 0:
            self.n_inputs = 0
            self.n_outputs = 0

    def eval(self, x):
        """Evaluates ensemble at given input x."""

        # NOTE that there is the potential drawback in this implementation that x is moved back and forth to the gpu

        assert self.n_models > 0, 'Ensemble is empty.'

        y = 0.0

        for model, copies in izip(self.models, self.n_copies):
            y += copies * model.eval(x)

        y /= self.n_models

        return y

    def eval_model(self, i, x):
        """Evaluates model i in the ensemble at given input x."""

        return self.models[i].eval(x)


class FastEnsemble:
    """
    Implements an ensemble of other models. Maintains only a single model, and a list of different parameter matrices.
    As a result, it is faster to create and more memory efficient, but slower to evaluate.
    """

    def __init__(self, model, copy=False):
        """Initializes the ensemble as empty."""

        self.model = deepcopy(model) if copy else model
        self.parms = []
        self.n_copies = []
        self.n_models = 0
        self.n_diff_models = 0
        self.n_inputs = model.n_inputs
        self.n_outputs = model.n_outputs

    def _load_model(self, i):
        """Loads parameters for model i."""

        for j, p in enumerate(self.parms[i]):
            self.model.parms[j].set_value(p)

    def add_new(self, parms, copy=False):
        """Adds a new set of parameters to the ensemble."""

        if copy:
            self.parms.append([x.get_value().copy() for x in parms])
        else:
            self.parms.append([x.get_value() for x in parms])
        self.n_copies.append(1)
        self.n_models += 1
        self.n_diff_models += 1

    def add_existing(self, i):
        """Adds an extra copy of model i in the ensemble."""

        self.n_copies[i] += 1
        self.n_models += 1

    def remove(self, i):
        """Removes a model at position i from the ensemble."""

        self.n_copies[i] -= 1

        if self.n_copies[i] == 0:
            del self.parms[i]
            del self.n_copies[i]
            self.n_diff_models -= 1

    def eval(self, x, mode='mean'):
        """Evaluates ensemble at given input x."""

        # NOTE that there is the potential drawback in this implementation that x is moved back and forth to the gpu

        assert self.n_models > 0, 'Ensemble is empty.'

        if mode == 'mean':

            y = 0.0

            for i, copies in enumerate(self.n_copies):
                self._load_model(i)
                y += copies * self.model.eval(x)

            y /= self.n_models

        elif mode == 'logmeanexp':

            y = []

            for i, copies in enumerate(self.n_copies):
                self._load_model(i)
                y.append(np.log(copies) + self.model.eval(x))

            y = logsumexp(np.array(y), axis=0) - np.log(self.n_models)

        else:
            raise ValueError('Unknown averaging mode.')

        return y

    def eval_model(self, i, x):
        """Evaluates model i in the ensemble at given input x."""

        self._load_model(i)
        return self.model.eval(x)

    def get_traces(self):
        """Returns matrices whose columns are traces of parameters, in the order they where added to the ensemble."""

        all_traces = []

        for i in xrange(len(self.model.parms)):

            traces = []

            for params, copies in izip(self.parms, self.n_copies):
                for n in xrange(copies):
                    traces.append(params[i].flatten())

            all_traces.append(np.array(traces))

        return all_traces