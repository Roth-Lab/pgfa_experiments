import h5py
import os
import numpy as np
import pandas as pd
import yaml

from pgfa.models.trace import TraceWriter
from pgfa.utils import Timer

import pgfa.feature_allocation_distributions
import pgfa.models.linear_gaussian
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

    data = load_data(args.data_file)

    model = get_model(
        data,
        config["model"]["K"]
    )

    model_updater = get_model_updater(config["sampler"][args.sampler_id], config["model"]["K"])

    model.params = load_params(args.params_file)

    model_updater._update_model_params(model)

    trace_writer = TraceWriter(args.trace_file, model)

    df = run(model, model_updater, trace_writer, time=config["run"]["max_time"])

    trace_writer.close()

    df["dataset"] = os.path.basename(args.data_file).split(".")[0]

    df["params_seed"] = os.path.splitext(os.path.basename(args.params_file))[0]

    df["restart_seed"] = args.seed

    df["sampler"] = args.sampler_id

    df.to_csv(args.out_file, compression="gzip", index=False, sep="\t")


def get_model(data, num_features):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(num_features)

    return pgfa.models.linear_gaussian.Model(data, feat_alloc_dist)


def get_model_updater(config, num_features):
    feat_alloc_updater_kwargs = config.get("kwargs", {})

    if num_features is None:
        feat_alloc_updater_kwargs["singletons_updater"] = pgfa.models.linear_gaussian.CollapsedSingletonsUpdater()

    else:
        feat_alloc_updater_kwargs["singletons_updater"] = None

    feat_alloc_updater = pgfa.utils.get_feat_alloc_updater(
        mixture_prob=config.get("mixture_prob", 0.0),
        updater=config["updater"],
        updater_kwargs=feat_alloc_updater_kwargs
    )

    return pgfa.models.linear_gaussian.ModelUpdater(feat_alloc_updater)


def load_data(file_name):
    with h5py.File(file_name, "r") as fh:
        data = fh["data"][()]

    return data


def load_params(file_name):
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

    return params


def run(model, model_updater, trace_writer, time=100):
    timer = Timer()

    trace = [_get_trace_row(model, 0.0)]

    while True:
        timer.start()

        model_updater.update(model)

        timer.stop()

        trace_writer.write_row(model, timer.elapsed)

        trace.append(_get_trace_row(model, timer.elapsed))

        if timer.elapsed >= time:
            break

    df = pd.DataFrame(trace)

    df = df[["time", "num_features", "num_features_used", "log_p", "rmse"]]

    return df


def compute_l2_error(data, params):
    data_pred = params.Z @ params.V

    return np.sqrt(np.mean(np.square(data_pred - data)))


def _get_trace_row(model, time):
    return {
        "log_p": model.log_p,
        "time": time,
        "num_features": model.params.K,
        "num_features_used": np.sum(np.sum(model.params.Z, axis=0) > 0),
        "rmse": compute_l2_error(model.data, model.params)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config-file", required=True)

    parser.add_argument("-d", "--data-file", required=True)

    parser.add_argument("-p", "--params-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("-t", "--trace-file", required=True)

    parser.add_argument("-s", "--sampler-id", required=True)

    parser.add_argument("--seed", default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
