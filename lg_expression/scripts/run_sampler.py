import h5py
import os
import numpy as np
import pandas as pd
import yaml

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

    data, data_true = load_data(args.data_file)

    model = get_model(
        data,
        config["model"]["K"]
    )

    model_updater = get_model_updater(config["sampler"][args.sampler_id], config["model"]["K"])

    df = run(data_true, model, model_updater, time=config["run"]["max_time"])

    df["dataset"] = os.path.basename(args.data_file).split(".")[0]

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
    df = pd.read_csv(file_name, index_col=0, sep="\t")

    data_true = df.T.values

    mask = np.random.uniform(0, 1, size=data_true.shape) <= 0.05

    data = data_true.copy()

    data[mask] = np.nan

    return data, data_true


def run(data_true, model, model_updater, time=100):
    timer = Timer()

    trace = [_get_trace_row(data_true, model, 0.0)]

    while True:
        timer.start()

        model_updater.update(model)

        timer.stop()

        trace.append(_get_trace_row(data_true, model, timer.elapsed))

        if timer.elapsed >= time:
            break

    df = pd.DataFrame(trace)

    df = df[["time", "num_features", "num_features_used", "log_p", "rmse"]]

    return df


def compute_l2_error(data, data_true, params):
    idxs = np.isnan(data)

    if not np.any(idxs):
        return 0

    data_pred = params.Z @ params.V

    return np.sqrt(np.mean(np.square(data_pred[idxs] - data_true[idxs])))


def _get_trace_row(data_true, model, time):
    return {
        "log_p": model.log_p,
        "time": time,
        "num_features": model.params.K,
        "num_features_used": np.sum(np.sum(model.params.Z, axis=0) > 0),
        "rmse": compute_l2_error(model.data, data_true, model.params)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config-file", required=True)

    parser.add_argument("-d", "--data-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("-s", "--sampler-id", required=True)

    parser.add_argument("--seed", default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
