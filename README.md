# Bayesian neural networks demo

A demo of Bayesian neural networks on a toy binary classification problem. 

Two ways of implementing Bayesian neural networks are demonstrated:
- Stochastic Variational Inference using local reparameterization [1]
- Hamiltonian Monte Carlo [2]

[1] Kingma et al., _Variational Dropout and the Local Reparameterization Trick_, NIPS 2015. [[paper]](https://arxiv.org/abs/1506.02557)

[2] Neal, _MCMC using Hamiltonian dynamics_, Handbook of Markov Chain Monte Carlo, 2011. [[paper]](https://arxiv.org/abs/1206.1901)

## How to run the code

Display the training data:
```
python run.py --show
```

Train a non-Bayesian neural network by minimizing cross-entropy:
```
python run.py --mle
```

Train a Bayesian neural network using Stochastic Variarional Inference:
```
python run.py --svi
```

Train a Bayesian neural network using Hamiltonian Monte Carlo:
```
python run.py --hmc
```

