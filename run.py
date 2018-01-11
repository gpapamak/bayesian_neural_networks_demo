import argparse
import bnn_demo


def parse_args():
    """
    Returns an object describing the command line.
    """

    parser = argparse.ArgumentParser(description='Bayesian neural networks demo.')
    group = parser.add_mutually_exclusive_group()

    group.add_argument('--show', action='store_true', help='show the dataset')
    group.add_argument('--mle', action='store_true', help='train a non-bayesian net using maximum likelihood')
    group.add_argument('--svi', action='store_true', help='train a bayesian net using stochastic variational inference')
    group.add_argument('--hmc', action='store_true', help='train a bayesian net using hamiltonian monte carlo')

    return parser.parse_args()


def main():

    args = parse_args()

    if args.show:
        bnn_demo.show_train_data()

    elif args.mle:
        bnn_demo.fit_neural_net_demo()

    elif args.svi:
        bnn_demo.bayesian_neural_net_svi_demo()

    elif args.hmc:
        bnn_demo.bayesian_neural_net_hmc_demo()

    else:
        print('No action specified.')


if __name__ == '__main__':
    main()
