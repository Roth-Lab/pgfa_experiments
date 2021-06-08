import h5py
import numpy as np
import pandas as pd
import os
import yaml

from pgfa.utils import Timer

import pgfa.feature_allocation_distributions
import pgfa.models.lfrm
import pgfa.models.linear_gaussian
import pgfa.models.pyclone.binomial
import pgfa.updates
import pgfa.utils


os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def main(args):
    with open(args.config_file, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    if args.seed is not None:
        pgfa.utils.set_seed(args.seed)

    if config["model_class"] == "pyclone":
        data, params_true = load_pyclone_data_file(args.data_file)

    else:
        with h5py.File(args.data_file, "r") as fh:
            data = fh["data"][()]

        params_true = load_params(config, args.data_file)

    model = get_model(
        config,
        data,
        args.params_file
    )

    model_updater = get_model_updater(config, args.num_particles)

    df = run(model, model_updater, params_true, time=config["run"]["max_time"])

    df["model_class"] = config["model_class"]

    df["data_seed"] = os.path.splitext(os.path.basename(args.data_file))[0]

    df["params_seed"] = os.path.splitext(os.path.basename(args.params_file))[0]

    df["restart_seed"] = args.seed

    df["num_particles"] = args.num_particles

    df.to_csv(args.out_file, compression="gzip", index=False, sep="\t")


def get_model(config, data, params_file):
    params = load_params(config, params_file)

    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(config["model"]["K"])

    if config["model_class"] == "lfrm":
        model = pgfa.models.lfrm.Model(data, feat_alloc_dist, params=params, symmetric=False)

    elif config["model_class"] == "lg":
        model = pgfa.models.linear_gaussian.Model(data, feat_alloc_dist, params=params)

    elif config["model_class"] == "pyclone":
        model = pgfa.models.pyclone.binomial.Model(data, feat_alloc_dist, params=params)

    return model


def get_model_updater(config, num_particles):
    feat_alloc_updater = pgfa.updates.DiscreteParticleFilterUpdater(
        annealing_power=1.0,
        conditional_update=True,
        num_particles=num_particles,
        singletons_updater=None,
        test_path="zeros"
    )

    if config["model_class"] == "lfrm":
        updater = pgfa.models.lfrm.ModelUpdater(feat_alloc_updater)

    elif config["model_class"] == "lg":
        updater = pgfa.models.linear_gaussian.ModelUpdater(feat_alloc_updater)

    elif config["model_class"] == "pyclone":
        updater = pgfa.models.pyclone.binomial.ModelUpdater(feat_alloc_updater)

    return updater


def load_params(config, file_name):
    if config["model_class"] == "lfrm":
        with h5py.File(file_name, "r") as fh:
            params = pgfa.models.lfrm.Parameters(
                fh["alpha"][()],
                np.ones(2),
                fh["tau"][()],
                np.ones(2),
                fh["V"][()],
                fh["Z"][()]
            )

    elif config["model_class"] == "lg":
        with h5py.File(file_name, "r") as fh:
            params = pgfa.models.linear_gaussian.Parameters(
                fh["alpha"][()],
                np.ones(2),
                fh["tau_v"][()],
                np.ones(2),
                fh["tau_x"][()],
                np.ones(2),
                fh["V"][()],
                fh["Z"][()]
            )

    elif config["model_class"] == "pyclone":
        with h5py.File(file_name, "r") as fh:
            params = pgfa.models.pyclone.binomial.Parameters(
                fh["alpha"][()],
                np.ones(2),
                fh["V"][()],
                np.ones(2),
                fh["Z"][()]
            )

    return params


def run(model, model_updater, params_true, time=100):
    timer = Timer()

    trace = [_get_trace_row(model, model_updater, params_true, 0.0)]

    while timer.elapsed < time:
        timer.start()

        model_updater.update(model)

        timer.stop()

        trace.append(_get_trace_row(model, model_updater, params_true, timer.elapsed))

    df = pd.DataFrame(trace)

    model.params = params_true

    df["log_p_true"] = model.log_p

    df["rel_log_p"] = (df["log_p"] - df["log_p_true"]) / df["log_p_true"].abs()

    df = df[["time", "num_features", "log_p", "log_p_true", "rel_log_p",
             "b_cubed_f", "b_cubed_p", "b_cubed_r", "max_particles"]]

    return df


def _get_trace_row(model, model_updater, params_true, time):
    f, p, r = pgfa.utils.get_b_cubed_score(params_true.Z, model.params.Z)

    return {
        "log_p": model.log_p,
        "time": time,
        "num_features": model.params.K,
        "b_cubed_f": f,
        "b_cubed_p": p,
        "b_cubed_r": r,
        "max_particles": model_updater.feat_alloc_updater.row_updater.max_particles

    }


def load_pyclone_data_file(file_name):
    with pd.HDFStore(file_name, "r") as store:
        data_df = store["data"]

        alpha = store["alpha"].values[0]

        V = store["V"].values

        Z = store["Z"].values

    params = pgfa.models.pyclone.binomial.Parameters(alpha, np.ones(2), V, np.ones(2), Z)

    samples = sorted(data_df["sample_id"].unique())

    data = []

    for _, mut_df in data_df.groupby("mutation_id"):
        sample_data_points = []

        for sample in samples:
            row = mut_df[mut_df["sample_id"] == sample].iloc[0]

            sample_data_points.append(
                pgfa.models.pyclone.utils.get_sample_data_point(
                    row["a"].astype(int),
                    row["b"].astype(int),
                    row["cn_major"].astype(int),
                    row["cn_minor"].astype(int),
                    row["cn_normal"].astype(int),
                    1e-3,
                    row["tumour_content"]
                )
            )

        data.append(pgfa.models.pyclone.utils.DataPoint(sample_data_points))

    return data, params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config-file", required=True)

    parser.add_argument("-d", "--data-file", required=True)

    parser.add_argument("-p", "--params-file", default=None)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("-n", "--num-particles", required=True, type=int)

    parser.add_argument("--seed", default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
