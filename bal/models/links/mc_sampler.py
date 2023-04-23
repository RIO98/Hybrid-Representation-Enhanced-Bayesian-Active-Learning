from __future__ import absolute_import

import warnings
from functools import partial
from itertools import chain
from itertools import starmap

import six
import torch
import torch.nn as nn

from .classifier import get_values


class MCSampler(nn.Module):
    """ Monte Carlo estimation to approximate the predictive distribution.
    Predictive variance is a metric indicating uncertainty.

    Args:
        predictor (~chainer.Link): Predictor network.
        mc_iteration (int): Number of iterations in MCMC sampling
        activation (list or callable, optional): Activation function. If the predictor makes multiple outputs,
            this must be a list of activations. Defaults to partial(F.softmax, axis=1).
        reduce_mean (list or callable, optional): Reduce function for mean tensor. If the predictor makes multiple outputs,
            this must be a list of callable functions. Defaults to partial(F.argmax, axis=1).
        reduce_var (list or callable, optional): Reduce function for variance tensor. If the predictor makes multiple outputs,
            this must be a list of callable functions. Defaults to partial(F.mean, axis=1).
        mode (str, optional): Method for calculating uncertainty. Defaults to 'variance'. (one of `{'variance', 'entropy'}`).
        eps (float, optional): Epsilon value for numerical stability. Default to 1e-8.
        x_keys (tuple, int or str, optional): Key to specify input variable from arguments.
            When it is ``int``, a variable in positional arguments is used.
            And when it is ``str``, a variable in keyword arguments is used.
            If you use multiple variables, please specify ``tuple`` of ``int`` or ``str``. Defaults to (0).

    See also: https://arxiv.org/pdf/1506.02142.pdf
              https://arxiv.org/pdf/1511.02680.pdf
    """

    def __init__(self,
                 predictor,
                 mc_iteration,
                 activation=partial(torch.softmax, dim=1),
                 reduce_mean=partial(torch.argmax, dim=1),
                 reduce_var=partial(torch.mean, dim=1),
                 mode='variance',
                 eps=1e-8,
                 x_keys=0,
                 return_all_latent=False
                 ):
        super(MCSampler, self).__init__()

        assert callable(predictor), 'predictor should be callable..'

        self.add_module('predictor', predictor)

        self.activation = activation
        self.mc_iteration = mc_iteration
        self.reduce_mean = reduce_mean
        self.reduce_var = reduce_var
        self.mode = mode
        self.eps = eps
        self.return_all_latent = return_all_latent

        self.x_keys = x_keys

    @staticmethod
    def concat_variables(arrays):
        return torch.cat([array[None] for array in arrays], dim=0)

    @staticmethod
    def concat_samples(samples):
        first_elem = samples[0]

        if isinstance(first_elem, (tuple, list)):
            result = []

            for i in six.moves.range(len(first_elem)):
                result.append(MCSampler.concat_variables(
                    [example[i] for example in samples]))

            return list(result)

        else:
            return MCSampler.concat_variables(samples)

    @staticmethod
    def predict(samples, mode='variance',
                reduce_mean=None, reduce_var=None,
                eps=1e-8):
        mean = torch.mean(samples, dim=0)

        if mode == 'variance':
            var = torch.var(samples, dim=0)
        elif mode == 'entropy':
            var = - mean * torch.log2(mean + eps)
        else:
            raise NotImplementedError('unsupported mode..')

        if reduce_mean is not None:
            mean = reduce_mean(mean)

        if reduce_var is not None:
            var = reduce_var(var)

        return mean, var

    def forward(self, *args, **kwargs):

        if self.training:
            warnings.warn('During the training phase, MCMC sampling is not executed..')
            return self.predictor(*args, **kwargs)

        x = get_values(args, kwargs, self.x_keys)

        # MCMC sampling
        mc_samples = []
        activation = self.activation

        for _ in range(self.mc_iteration):

            if self.return_all_latent:
                logits, stored_activations = self.predictor(x)
            else:
                logits = self.predictor(x)

            if activation is None:
                y = logits
            elif isinstance(logits, (list, tuple)):
                assert isinstance(activation, (list, tuple))
                assert len(logits) == len(activation)

                y = list(starmap(lambda f, x: f(x), zip(activation, logits)))
            else:
                y = activation(logits)

            mc_samples.append(y)

        mc_samples = self.concat_samples(mc_samples)

        # uncertainty estimates
        reduce_mean = self.reduce_mean
        reduce_var = self.reduce_var

        if isinstance(mc_samples, list):
            if reduce_mean is None:
                reduce_mean = [None] * len(mc_samples)
            if reduce_var is None:
                reduce_var = [None] * len(mc_samples)

            assert isinstance(reduce_mean, (list, tuple))
            assert isinstance(reduce_var, (list, tuple))

            ret = list(starmap(lambda _samples, _reduce_m, _reduce_v:
                               self.predict(_samples, self.mode,
                                            _reduce_m, _reduce_v,
                                            self.eps),
                               zip(mc_samples, reduce_mean, reduce_var)))

            ret = list(chain.from_iterable(ret))

        else:
            ret = self.predict(mc_samples, self.mode,
                               self.reduce_mean, self.reduce_var,
                               self.eps)

        if self.return_all_latent:
            return ret, stored_activations

        return ret
