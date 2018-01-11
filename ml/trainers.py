from itertools import izip

import os
import sys
import numpy as np
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import ml.step_strategies as ss
import ml.data_streams as ds
import ml.ensembles as ensembles
import util.math


dtype = theano.config.floatX


class SGD:
    """
    Minibatch stochastic gradient descent. Can work with a variety of step strategies, and supports early stopping on
    validation set.
    """

    def __init__(self, model, trn_data, trn_loss, trn_target=None, val_data=None, val_loss=None, val_target=None, step=ss.Adam()):
        """
        Constructs and configures the trainer.
        :param model: the model to be trained
        :param trn_data: train inputs and (possibly) train targets
        :param trn_loss: theano variable representing the train loss to minimize
        :param trn_target: theano variable representing the train target
        :param val_data: validation inputs and (possibly) validation targets
        :param val_loss: theano variable representing the validation loss
        :param val_target: theano variable representing the validation target
        :param step: step size strategy object
        :return: None
        """

        # parse input
        # TODO: it would be good to type check the other inputs too
        assert isinstance(step, ss.StepStrategy), 'Step must be a step strategy object.'

        # prepare train data
        n_trn_data_list = set([x.shape[0] for x in trn_data])
        assert len(n_trn_data_list) == 1, 'Number of train data is not consistent.'
        self.n_trn_data = list(n_trn_data_list)[0]
        trn_data = [theano.shared(x.astype(dtype), borrow=True) for x in trn_data]

        # compile theano function for a single training update
        grads = tt.grad(trn_loss, model.parms)
        idx = tt.ivector('idx')
        trn_inputs = [model.input] if trn_target is None else [model.input, trn_target]
        self.make_update = theano.function(
            inputs=[idx],
            outputs=trn_loss,
            givens=zip(trn_inputs, [x[idx] for x in trn_data]),
            updates=step.updates(model.parms, grads)
        )

        # if model uses batch norm, compile a theano function for setting up stats
        if getattr(model, 'batch_norm', False):
            batch_norm_givens = [(bn.m, bn.bm) for bn in model.bns] + [(bn.v, bn.bv) for bn in model.bns]
            self.set_batch_norm_stats = theano.function(
                inputs=[],
                givens=zip(trn_inputs, trn_data),
                updates=[(bn.bm, bn.m) for bn in model.bns] + [(bn.bv, bn.v) for bn in model.bns]
            )
        else:
            self.set_batch_norm_stats = None
            batch_norm_givens = []

        # if validation data is given, then set up validation too
        self.do_validation = val_data is not None

        if self.do_validation:

            # prepare validation data
            n_val_data_list = set([x.shape[0] for x in val_data])
            assert len(n_val_data_list) == 1, 'Number of validation data is not consistent.'
            self.n_val_data = list(n_val_data_list)[0]
            val_data = [theano.shared(x.astype(dtype), borrow=True) for x in val_data]

            # compile theano function for validation
            val_inputs = [model.input] if val_target is None else [model.input, val_target]
            self.validate = theano.function(
                inputs=[],
                outputs=val_loss,
                givens=zip(val_inputs, val_data) + batch_norm_givens
            )

            # create checkpointer to store best model
            self.checkpointer = ModelCheckpointer(model)
            self.best_val_loss = float('inf')

        # initialize some variables
        self.trn_loss = float('inf')
        self.idx_stream = ds.IndexSubSampler(self.n_trn_data, rng=np.random.RandomState(42))

    def train(self, minibatch=None, tol=None, maxepochs=None, monitor_every=None, patience=None, logger=sys.stdout, show_progress=False, val_in_same_plot=True):
        """
        Trains the model.
        :param minibatch: minibatch size
        :param tol: tolerance
        :param maxepochs: maximum number of epochs
        :param monitor_every: monitoring frequency
        :param patience: maximum number of validation steps to wait for improvement before early stopping
        :param logger: logger for logging messages. If None, no logging takes place
        :param show_progress: if True, plot training and validation progress
        :param val_in_same_plot: if True, plot validation progress in same plot as training progress
        :return: None
        """

        # parse input
        assert minibatch is None or util.math.isposint(minibatch), 'Minibatch size must be a positive integer or None.'
        assert tol is None or tol > 0.0, 'Tolerance must be positive or None.'
        assert maxepochs is None or maxepochs > 0.0, 'Maximum number of epochs must be positive or None.'
        assert monitor_every is None or monitor_every > 0.0, 'Monitoring frequency must be positive or None.'
        assert patience is None or util.math.isposint(patience), 'Patience must be a positive integer or None.'
        assert isinstance(show_progress, bool), 'store_progress must be boolean.'
        assert isinstance(val_in_same_plot, bool), 'val_in_same_plot must be boolean.'

        # initialize some variables
        iter = 0
        progress_epc = []
        progress_trn = []
        progress_val = []
        minibatch = self.n_trn_data if minibatch is None else minibatch
        maxiter = float('inf') if maxepochs is None else np.ceil(maxepochs * self.n_trn_data / float(minibatch))
        monitor_every = float('inf') if monitor_every is None else np.ceil(monitor_every * self.n_trn_data / float(minibatch))
        patience = float('inf') if patience is None else patience
        patience_left = patience
        best_epoch = None
        logger = open(os.devnull, 'w') if logger is None else logger

        # main training loop
        while True:

            # make update to parameters
            trn_loss = self.make_update(self.idx_stream.gen(minibatch))
            diff = self.trn_loss - trn_loss
            iter += 1
            self.trn_loss = trn_loss

            if iter % monitor_every == 0:

                epoch = iter * float(minibatch) / self.n_trn_data

                # do validation
                if self.do_validation:
                    if self.set_batch_norm_stats is not None: self.set_batch_norm_stats()
                    val_loss = self.validate()
                    patience_left -= 1

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.checkpointer.checkpoint()
                        best_epoch = epoch
                        patience_left = patience

                # monitor progress
                if show_progress:
                    progress_epc.append(epoch)
                    progress_trn.append(trn_loss)
                    if self.do_validation: progress_val.append(val_loss)

                # log info
                if self.do_validation:
                    logger.write('Epoch = {0:.2f}, train loss = {1}, validation loss = {2}\n'.format(epoch, trn_loss, val_loss))
                else:
                    logger.write('Epoch = {0:.2f}, train loss = {1}\n'.format(epoch, trn_loss))

            # check for convergence
            if abs(diff) < tol or iter >= maxiter or patience_left <= 0:
                if self.do_validation: self.checkpointer.restore()
                if self.set_batch_norm_stats is not None: self.set_batch_norm_stats()
                break

        # plot progress
        if show_progress:

            if self.do_validation:

                if val_in_same_plot:
                    fig, ax = plt.subplots(1, 1)
                    ax.semilogx(progress_epc, progress_trn, 'b', label='training')
                    ax.semilogx(progress_epc, progress_val, 'r', label='validation')
                    ax.vlines(best_epoch, ax.get_ylim()[0], ax.get_ylim()[1], color='g', linestyles='dashed', label='best')
                    ax.set_xlabel('epochs')
                    ax.set_ylabel('loss')
                    ax.legend()
                    ax.set_title('Training progress')

                else:
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                    ax1.semilogx(progress_epc, progress_trn, 'b')
                    ax2.semilogx(progress_epc, progress_val, 'r')
                    ax1.vlines(best_epoch, ax1.get_ylim()[0], ax1.get_ylim()[1], color='g', linestyles='dashed', label='best')
                    ax2.vlines(best_epoch, ax2.get_ylim()[0], ax2.get_ylim()[1], color='g', linestyles='dashed', label='best')
                    ax2.set_xlabel('epochs')
                    ax1.set_ylabel('training loss')
                    ax2.set_ylabel('validation loss')
                    fig.suptitle('Training progress')

            else:
                fig, ax = plt.subplots(1, 1)
                ax.semilogx(progress_epc, progress_trn, 'b')
                ax.set_xlabel('epochs')
                ax.set_ylabel('training loss')
                ax.legend()
                ax.set_title('Training progress')

            plt.show(block=False)


class HMC:
    """
    Hamiltonian Monte Carlo training of models. Uses a quadratic kinetic energy. Trained model is an ensemble of
    posterior model samples.
    """

    def __init__(self, model, trn_data, trn_loss, trn_target):
        """
        :param model: model to train
        :param trn_data: train data
        :param trn_loss: train loss
        :param trn_target: train target
        """

        # prepare train data
        n_trn_data_list = set([x.shape[0] for x in trn_data])
        assert len(n_trn_data_list) == 1, 'Number of train data is not consistent.'
        trn_data = [theano.shared(x.astype(dtype)) for x in trn_data]

        # prepare train inputs
        trn_inputs = [model.input] if trn_target is None else [model.input, trn_target]

        # potential energy
        self.U = theano.function(
            inputs=[],
            outputs=trn_loss,
            givens=zip(trn_inputs, trn_data)
        )

        # theano variables
        step = tt.scalar('step')
        mass = tt.scalar('mass')
        srng = RandomStreams()

        # theano function for drawing random momentum variables
        ps = [theano.shared(np.zeros_like(x.get_value(borrow=True)), borrow=True) for x in model.parms]
        ps_rand = [srng.normal(x.get_value().shape, std=tt.sqrt(mass), dtype=dtype) for x in model.parms]
        ps_rand = [tt.unbroadcast(pr, *range(x.get_value().ndim)) for pr, x in izip(ps_rand, model.parms)]
        self.draw_momentum = theano.function(
            inputs=[mass],
            updates=zip(ps, ps_rand),
            allow_input_downcast=True
        )

        # theano function for calculating kinetic energy
        K = sum([tt.sum(p**2) for p in ps]) / (2.0 * mass)
        self.calc_kinetic = theano.function(
            inputs=[mass],
            outputs=K,
            allow_input_downcast=True
        )

        # theano function for updating momentum variables
        dUs = tt.grad(trn_loss, model.parms)
        new_ps = [p - step * dU for p, dU in izip(ps, dUs)]
        self.update_momentum = theano.function(
            inputs=[step],
            updates=zip(ps, new_ps),
            givens=zip(trn_inputs, trn_data),
            allow_input_downcast=True
        )

        # theano function for updating model parameters
        new_parms = [x + step / mass * p for x, p in izip(model.parms, ps)]
        self.update_parms = theano.function(
            inputs=[step, mass],
            updates=zip(model.parms, new_parms),
            allow_input_downcast=True
        )

        # initialize
        self.U_prev = self.U()
        self.model = model

    def gen(self, n_samples, L, me, m=1.0, logger=sys.stdout, show_traces=False, rng=np.random):
        """
        Generates HMC samples.
        :param n_samples: number of samples
        :param L: number of leapfrog steps
        :param me: mean of time step
        :param m: mass
        :param logger: logger for logging messages. If None, no logging takes place
        :param show_traces: whether to plot info at the end of sampling
        :param rng: random number generator to use
        :return: an ensemble of model samples
        """

        # initialize
        n_acc = 0
        U_trace = []
        H_error_trace = []
        acc_rate_trace = []
        xs = self.model.parms
        ensemble = ensembles.FastEnsemble(self.model, copy=True)
        ensemble.add_new(xs, copy=True)
        logger = open(os.devnull, 'w') if logger is None else logger

        for n in xrange(n_samples):

            # sample momentum from a gaussian
            self.draw_momentum(m)
            K_prev = self.calc_kinetic(m)

            # simulate hamiltonian dynamics with leapfrog method
            e = -me * np.log(1 - rng.rand())
            self.update_momentum(0.5 * e)
            for _ in xrange(L-1):
                self.update_parms(e, m)
                self.update_momentum(e)
            self.update_parms(e, m)
            self.update_momentum(0.5 * e)
            # negating p is not necessary, because kinetic energy is symmetric

            # metropolis acceptance rule
            U_new = self.U()
            K_new = self.calc_kinetic(m)
            H_err = (U_new + K_new) - (self.U_prev + K_prev)
            if rng.rand() < np.exp(-H_err):
                self.U_prev = U_new
                n_acc += 1
                ensemble.add_new(xs, copy=True)
            else:
                for i, x in enumerate(ensemble.parms[-1]):
                    xs[i].set_value(x.copy())
                ensemble.add_existing(-1)

            # acceptance rate
            acc_rate = n_acc / float(n+1)
            logger.write('sample = {0}, acc rate = {1:.2%}, hamiltonian error = {2:.2}\n'.format(n+1, acc_rate, H_err))

            # record traces
            if show_traces:
                U_trace.append(self.U_prev)
                H_error_trace.append(H_err)
                acc_rate_trace.append(acc_rate)

        ensemble.remove(0)

        # show plot with the traces
        if show_traces:

            fig, ax = plt.subplots(3, 1, sharex=True)
            ax[0].plot(U_trace)
            ax[0].set_ylabel('potential energy')
            ax[1].plot(H_error_trace)
            ax[1].set_ylabel('hamiltonian error')
            ax[2].plot(acc_rate_trace)
            ax[2].set_ylim([0, 1])
            ax[2].set_ylabel('acceptance rate')
            ax[2].set_xlabel('samples')
            fig.suptitle('HMC progress')

            parm_traces = ensemble.get_traces()
            fig, axs = plt.subplots(len(parm_traces), sharex=True)
            for ax, p in izip(axs, parm_traces):
                ax.plot(p)
            axs[-1].set_xlabel('samples')
            fig.suptitle('Parameter traces')

            plt.show(block=False)

        return ensemble


class ModelCheckpointer:
    """
    Helper class which makes checkpoints of a given model.
    Currently one checkpoint is supported; checkpointing twice overwrites previous checkpoint.
    """

    def __init__(self, model):
        """
        :param model: A machine learning model to be checkpointed.
        """
        self.model = model
        self.checkpointed_parms = [np.empty_like(p.get_value()) for p in model.parms]

    def checkpoint(self):
        """
        Checkpoints current model. Overwrites previous checkpoint.
        """
        for i, p in enumerate(self.model.parms):
            self.checkpointed_parms[i] = p.get_value().copy()

    def restore(self):
        """
        Restores last checkpointed model.
        """
        for i, p in enumerate(self.checkpointed_parms):
            self.model.parms[i].set_value(p)
