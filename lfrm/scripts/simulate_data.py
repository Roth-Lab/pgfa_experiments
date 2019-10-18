import h5py
import numpy as np
import pgfa.models.lfrm
import yaml


def main(args):
    with open(args.config_file, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)['data']

    if args.seed is not None:
        np.random.seed(args.seed)

    params = pgfa.models.lfrm.simulate_params(
        alpha=config['alpha'],
        tau=config['tau'],
        K=config['K'],
        N=config['N']
    )

    data, data_true = pgfa.models.lfrm.simulate_data(
        params, prop_missing=config['prop_missing'], symmetric=config['symmetric']
    )

    with h5py.File(args.out_file, 'w') as fh:
        fh.create_dataset('data', compression='gzip', data=data)

        fh.create_dataset('data_true', compression='gzip', data=data_true)

        fh.create_dataset('V', compression='gzip', data=params.V)

        fh.create_dataset('Z', compression='gzip', data=params.Z)

        fh.create_dataset('alpha', data=params.alpha)

        fh.create_dataset('tau', data=params.tau)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config-file', required=True)

    parser.add_argument('-o', '--out-file', required=True)

    parser.add_argument('--seed', default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
