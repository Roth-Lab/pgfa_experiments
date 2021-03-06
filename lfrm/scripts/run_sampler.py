import os
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import h5py
import numpy as np
import pandas as pd
import yaml

from pgfa.utils import Timer

import pgfa.feature_allocation_distributions
import pgfa.models.lfrm
import pgfa.utils


def main(args):
    with open(args.config_file, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    if args.seed is not None:
        pgfa.utils.set_seed(args.seed)

    with h5py.File(args.data_file, 'r') as fh:
        data_true = fh['data_true'][()]

    params_true = load_params(args.data_file)

    model = get_model(
        args.data_file,
        config['model']['K'],
        params_file=args.params_file,
        symmetric=config['model']['symmetric']
    )

    model_updater = get_model_updater(config['sampler'][args.sampler_id], config['model']['K'])

    df = run(data_true, model, model_updater, params_true, time=config['run']['max_time'])

    df['data_seed'] = os.path.splitext(os.path.basename(args.data_file))[0]

    df['params_seed'] = os.path.splitext(os.path.basename(args.params_file))[0]

    df['restart_seed'] = args.seed

    df['sampler'] = args.sampler_id

    df.to_csv(args.out_file, compression='gzip', index=False, sep='\t')


def get_model(data_file, num_features, params_file=None, symmetric=False):
    with h5py.File(data_file, 'r') as fh:
        data = fh['data'][()]

    if params_file is not None:
        params = load_params(params_file)

    else:
        params = None

    feat_alloc_dist = pgfa.feature_allocation_distributions.get_feature_allocation_distribution(num_features)

    return pgfa.models.lfrm.Model(data, feat_alloc_dist, params=params, symmetric=symmetric)


def get_model_updater(config, num_features):
    feat_alloc_updater_kwargs = config.get('kwargs', {})

    if num_features is None:
        feat_alloc_updater_kwargs['singletons_updater'] = pgfa.models.lfrm.PriorSingletonsUpdater()

    else:
        feat_alloc_updater_kwargs['singletons_updater'] = None

    feat_alloc_updater = pgfa.utils.get_feat_alloc_updater(
        mixture_prob=config.get('mixture_prob', 0.0),
        updater=config['updater'],
        updater_kwargs=feat_alloc_updater_kwargs
    )

    return pgfa.models.lfrm.ModelUpdater(feat_alloc_updater)


def load_params(file_name):
    with h5py.File(file_name, 'r') as fh:
        params = pgfa.models.lfrm.Parameters(
            fh['alpha'][()],
            np.ones(2),
            fh['tau'][()],
            np.ones(2),
            fh['V'][()],
            fh['Z'][()]
        )

    return params


def run(data_true, model, model_updater, params_true, time=100):
    timer = Timer()

    trace = [_get_trace_row(data_true, model, params_true, 0.0)]

    while timer.elapsed < time:
        timer.start()

        model_updater.update(model)

        timer.stop()

        trace.append(_get_trace_row(data_true, model, params_true, timer.elapsed))

    df = pd.DataFrame(trace)

    model.params = params_true

    df['log_p_true'] = model.log_p

    df['rel_log_p'] = (df['log_p'] - df['log_p_true']) / df['log_p_true'].abs()

    df = df[['time', 'num_features', 'log_p', 'log_p_true', 'rel_log_p', 'b_cubed_f', 'b_cubed_p', 'b_cubed_r', 'error']]

    return df


def _get_trace_row(data_true, model, params_true, time):
    f, p, r = pgfa.utils.get_b_cubed_score(params_true.Z, model.params.Z)

    return {
        'log_p': model.log_p,
        'time': time,
        'num_features': model.params.K,
        'b_cubed_f': f,
        'b_cubed_p': p,
        'b_cubed_r': r,
        'error': np.sum(np.abs(model.predict(method='max') - data_true))
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config-file', required=True)

    parser.add_argument('-d', '--data-file', required=True)

    parser.add_argument('-p', '--params-file', default=None)

    parser.add_argument('-o', '--out-file', required=True)

    parser.add_argument('-s', '--sampler-id', required=True)

    parser.add_argument('--seed', default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
