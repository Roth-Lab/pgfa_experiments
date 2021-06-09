from sklearn.mixture import GaussianMixture

import h5py
import numpy as np
import pandas as pd
import yaml


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)

    with open(args.config_file, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    df = pd.read_csv(args.data_file, index_col=[0, 1], sep="\t")

    D, N = df.shape

    K = config["model"]["K"]

    gmm = GaussianMixture(n_components=K)

    gmm.fit(df.T.values)

    labels = gmm.predict(df.T.values)

    Z = np.zeros((N, K), dtype=int)

    for i in range(N):
        Z[i, labels[i]] = 1

    V = np.zeros((K, D))

    for k in range(K):
        V[k] = gmm.means_[k]

    with h5py.File(args.out_file, 'w') as fh:
        fh.create_dataset('V', compression='gzip', data=V)

        fh.create_dataset('Z', compression='gzip', data=Z)

        fh.create_dataset('alpha', data=1)

        fh.create_dataset('tau_v', data=1)

        fh.create_dataset('tau_x', data=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config-file", required=True)

    parser.add_argument("-d", "--data-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("--seed", default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
