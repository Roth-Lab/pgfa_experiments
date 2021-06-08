import h5py
import numpy as np
import pandas as pd
import scipy
import yaml

import pgfa.models.lfrm
import pgfa.models.linear_gaussian
import pgfa.models.pyclone.binomial


def main(args):
    with open(args.config_file, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    if args.seed is not None:
        np.random.seed(args.seed)

    if config["model_class"] == "lfrm":
        params = pgfa.models.lfrm.simulate_params(
            alpha=config["data"]["alpha"],
            tau=config["data"]["tau"],
            K=config["data"]["K"],
            N=config["data"]["N"]
        )

        data, data_true = pgfa.models.lfrm.simulate_data(
            params, prop_missing=config["data"]["prop_missing"], symmetric=config["data"]["symmetric"]
        )

        with h5py.File(args.out_file, "w") as fh:
            fh.create_dataset("data", compression="gzip", data=data)

            fh.create_dataset("data_true", compression="gzip", data=data_true)

            fh.create_dataset("V", compression="gzip", data=params.V)

            fh.create_dataset("Z", compression="gzip", data=params.Z)

            fh.create_dataset("alpha", data=params.alpha)

            fh.create_dataset("tau", data=params.tau)

    elif config["model_class"] == "lg":
        params = pgfa.models.linear_gaussian.simulate_params(
            alpha=config["data"]["alpha"],
            tau_v=config["data"]["tau_v"],
            tau_x=config["data"]["tau_x"],
            D=config["data"]["D"],
            K=config["data"]["K"],
            N=config["data"]["N"]
        )

        data, data_true = pgfa.models.linear_gaussian.simulate_data(
            params, prop_missing=config["data"]["prop_missing"])

        with h5py.File(args.out_file, "w") as fh:
            fh.create_dataset("data", compression="gzip", data=data)

            fh.create_dataset("data_true", compression="gzip", data=data_true)

            fh.create_dataset("V", compression="gzip", data=params.V)

            fh.create_dataset("Z", compression="gzip", data=params.Z)

            fh.create_dataset("alpha", data=params.alpha)

            fh.create_dataset("tau_v", data=params.tau_v)

            fh.create_dataset("tau_x", data=params.tau_x)

    elif config["model_class"] == "pyclone":
        params = pgfa.models.pyclone.binomial.simulate_params(
            alpha=config["data"]["alpha"],
            D=config["data"]["D"],
            K=config["data"]["K"],
            N=config["data"]["N"]
        )

        data = simulate_pyclone_data(params, eps=1e-3)

        with pd.HDFStore(args.out_file, "w") as store:
            store["data"] = data

            store["alpha"] = pd.Series([params.alpha])

            store["V"] = pd.DataFrame(params.V)

            store["Z"] = pd.DataFrame(params.Z)


def simulate_pyclone_data(params, eps=1e-3):
    F = params.F

    data = []

    cn_n = 2

    cn_r = 2

    mu_n = eps

    mu_r = eps

    t = np.ones(params.D)

    for n in range(params.N):
        phi = params.Z[n] @ F

        cn_total = 2

        cn_major = scipy.stats.randint.rvs(1, cn_total + 1)

        cn_minor = cn_total - cn_major

        cn_var = scipy.stats.randint.rvs(1, cn_major + 1)

        for s in range(params.D):
            mu_v = min(cn_var / cn_total, 1 - eps)

            xi = (1 - t[s]) * phi[s] * cn_n * mu_n + t[s] * (1 - phi[s]) * \
                cn_r * mu_r + t[s] * phi[s] * cn_total * mu_v

            xi /= (1 - t[s]) * phi[s] * cn_n + t[s] * (1 - phi[s]) * cn_r + t[s] * phi[s] * cn_total

            d = scipy.stats.poisson.rvs(1000)

            b = scipy.stats.binom.rvs(d, xi)

            a = d - b

            data.append(
                {
                    'mutation_id': n,
                    'sample_id': s,
                    'a': a,
                    'b': b,
                    'cn_normal': cn_n,
                    'cn_major': cn_major,
                    'cn_minor': cn_minor,
                    'tumour_content': t[s]
                }
            )

    data = pd.DataFrame(data)

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("--seed", default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
