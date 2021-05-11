import os
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import h5py
import pandas as pd
import numpy as np
import yaml

from pgfa.utils import Timer

import pgfa.utils
import pgfa.models.pyclone.singletons_updates
import pgfa.models.pyclone.binomial
import pgfa.feature_allocation_distributions


def main(args):
    with open(args.config_file, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    if args.seed is not None:
        pgfa.utils.set_seed(args.seed)

    data = load_data(args.data_file)

    model = get_model(
        data,
        config['model']['K']
    )

    model_updater = get_model_updater(config['sampler'][args.sampler_id], config['model']['K'])

    df = run(model, model_updater, time=config['run']['max_time'])

    df['restart_seed'] = args.seed

    df['sampler'] = args.sampler_id

    df.to_csv(args.out_file, compression='gzip', index=False, sep='\t')


def get_model(data, num_features):
    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(num_features)

    return pgfa.models.pyclone.binomial.Model(data, feat_alloc_dist)


def get_model_updater(config, num_features):
    feat_alloc_updater_kwargs = config.get('kwargs', {})

    if num_features is None:
        feat_alloc_updater_kwargs['singletons_updater'] = pgfa.models.pyclone.singletons_updates.PriorSingletonsUpdater()

    else:
        feat_alloc_updater_kwargs['singletons_updater'] = None

    feat_alloc_updater = pgfa.utils.get_feat_alloc_updater(
        mixture_prob=config.get('mixture_prob', 0.0),
        updater=config['updater'],
        updater_kwargs=feat_alloc_updater_kwargs
    )

    return pgfa.models.pyclone.binomial.ModelUpdater(feat_alloc_updater)


def load_data(file_name):
    df = pd.read_csv(file_name, sep="\t")

    samples = df["sample_id"].unique()

    df = df.groupby("mutation_id").filter(lambda x: x["sample_id"].nunique() == len(samples))

    data = []

    for mutation_id, group in df.groupby("mutation_id"):
        group = group.set_index("sample_id")

        mutation_data = []

        for sample_id in samples:
            row = group.loc[sample_id]

            sample_data = pgfa.models.pyclone.utils.get_sample_data_point(
                a=row["ref_counts"],
                b=row["alt_counts"],
                cn_major=row["major_cn"],
                cn_minor=row["minor_cn"],
                tumour_content=row["tumour_content"]
            )

            mutation_data.append(sample_data)

        data.append(pgfa.models.pyclone.utils.DataPoint(mutation_data))

    return data


def run(model, model_updater, time=100):
    timer = Timer()

    trace = [_get_trace_row(model, 0.0)]

    while timer.elapsed < time:
        timer.start()

        model_updater.update(model)

        timer.stop()

        trace.append(_get_trace_row(model,  timer.elapsed))

    df = pd.DataFrame(trace)

    df = df[['time', 'num_features', 'num_features_used', 'log_p']]

    return df


def _get_trace_row(model, time):
    return {
        'log_p': model.log_p,
        'time': time,
        'num_features': model.params.K,
        'num_features_used': np.sum(np.sum(model.params.Z, axis=0) > 0)
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config-file', required=True)

    parser.add_argument('-d', '--data-file', required=True)

    parser.add_argument('-o', '--out-file', required=True)

    parser.add_argument('-s', '--sampler-id', required=True)

    parser.add_argument('--seed', default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
