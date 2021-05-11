import h5py
import numpy as np
import yaml

import pgfa.models.lfrm


def main(args):
    with open(args.config_file, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    if args.seed is not None:
        np.random.seed(args.seed)

    params = pgfa.models.lfrm.simulate_params(
        alpha=1.0,
        tau=1.0,
        K=config['model']['K'],
        N=config['data']['N']
    )

    with h5py.File(args.out_file, 'w') as fh:
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
