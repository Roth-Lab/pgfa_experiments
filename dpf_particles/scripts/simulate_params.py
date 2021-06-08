import h5py
import numpy as np
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
            alpha=1.0,
            tau=1.0,
            K=config["model"]["K"],
            N=config["data"]["N"]
        )

        with h5py.File(args.out_file, "w") as fh:
            fh.create_dataset("V", compression="gzip", data=params.V)

            fh.create_dataset("Z", compression="gzip", data=params.Z)

            fh.create_dataset("alpha", data=params.alpha)

            fh.create_dataset("tau", data=params.tau)

    elif config["model_class"] == "lg":
        params = pgfa.models.linear_gaussian.simulate_params(
            alpha=1.0,
            tau_v=1.0,
            tau_x=1.0,
            D=config["data"]["D"],
            K=config["model"]["K"],
            N=config["data"]["N"]
        )

        with h5py.File(args.out_file, "w") as fh:
            fh.create_dataset("V", compression="gzip", data=params.V)

            fh.create_dataset("Z", compression="gzip", data=params.Z)

            fh.create_dataset("alpha", data=params.alpha)

            fh.create_dataset("tau_v", data=params.tau_v)

            fh.create_dataset("tau_x", data=params.tau_x)

    elif config["model_class"] == "pyclone":
        params = pgfa.models.pyclone.binomial.simulate_params(
            D=config["data"]["D"],
            K=config["model"]["K"],
            N=config["data"]["N"],
            alpha=1.0
        )

        with h5py.File(args.out_file, "w") as fh:
            fh.create_dataset("V", compression="gzip", data=params.V)

            fh.create_dataset("Z", compression="gzip", data=params.Z)

            fh.create_dataset("alpha", data=params.alpha)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("--seed", default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
