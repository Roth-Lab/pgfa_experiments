import h5py
import numpy as np
import os
import pandas as pd
import yaml

from pgfa.utils import Timer

import pgfa.feature_allocation_distributions
import pgfa.models.pyclone.binomial
import pgfa.models.pyclone.singletons_updates
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

    data, gt, num_samples = load_data(args.data_file, args.ground_truth_file)

    model = get_model(
        data,
        config["model"]["K"]
    )

    model_updater = get_model_updater(config["sampler"][args.sampler_id], config["model"]["K"])

    df = run(gt, model, model_updater, time=config["run"]["max_time"])

    df["dataset"] = args.dataset

    df["num_samples"] = num_samples

    df["data_file"] = os.path.basename(args.data_file)

    df["restart_seed"] = args.seed

    df["sampler"] = args.sampler_id

    df.to_csv(args.out_file, compression="gzip", index=False, sep="\t")


def get_model(data, num_features):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(num_features)

    return pgfa.models.pyclone.binomial.Model(data, feat_alloc_dist)


def get_model_updater(config, num_features):
    feat_alloc_updater_kwargs = config.get("kwargs", {})

    if num_features is None:
        feat_alloc_updater_kwargs["singletons_updater"] = pgfa.models.pyclone.singletons_updates.PriorSingletonsUpdater()

    else:
        feat_alloc_updater_kwargs["singletons_updater"] = None

    feat_alloc_updater = pgfa.utils.get_feat_alloc_updater(
        mixture_prob=config.get("mixture_prob", 0.0),
        updater=config["updater"],
        updater_kwargs=feat_alloc_updater_kwargs
    )

    return pgfa.models.pyclone.binomial.ModelUpdater(feat_alloc_updater)


def load_data(data_file, gt_file):
    gt = pd.read_csv(gt_file, header=None, sep="\t")

    gt = gt.iloc[:, 1:]

    df = pd.read_csv(data_file, sep="\t")

    samples = df["sample_id"].unique()

    # df = df.groupby("mutation_id").filter(lambda x: np.all(x["cn"] <= 2))

    df = df.groupby("mutation_id").filter(lambda x: x["sample_id"].nunique() == len(samples))

    data = []

    for _, row in df.iterrows():
        mutation_data = []

        sample_data = pgfa.models.pyclone.utils.get_sample_data_point(
            a=row["ref_counts"],
            b=row["alt_counts"],
            cn_major=row["major_cn"],
            cn_minor=row["minor_cn"],
            tumour_content=row["tumour_content"]
        )

        mutation_data.append(sample_data)

        data.append(pgfa.models.pyclone.utils.DataPoint(mutation_data))

    return data, gt, len(samples)


def run(gt, model, model_updater, time=100):
    timer = Timer()

    trace = [_get_trace_row(gt, model, 0.0)]

    while timer.elapsed < time:
        timer.start()

        model_updater.update(model)

        timer.stop()

        trace.append(_get_trace_row(gt, model,  timer.elapsed))

    df = pd.DataFrame(trace)

    df = df[["time", "num_features", "num_features_used", "log_p", "b_cubed_f", "b_cubed_p", "b_cubed_r"]]

    return df


def _get_trace_row(gt, model, time):
    f, p, r = pgfa.utils.get_b_cubed_score(gt.values, model.params.Z)

    return {
        "log_p": model.log_p,
        "time": time,
        "num_features": model.params.K,
        "num_features_used": np.sum(np.sum(model.params.Z, axis=0) > 0),
        'b_cubed_f': f,
        'b_cubed_p': p,
        'b_cubed_r': r
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config-file", required=True)

    parser.add_argument("-d", "--data-file", required=True)

    parser.add_argument("-g", "--ground-truth-file", required=True)

    parser.add_argument("-o", "--out-file", required=True)

    parser.add_argument("-s", "--sampler-id", required=True)

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--seed", default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
