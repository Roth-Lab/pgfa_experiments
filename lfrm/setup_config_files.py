import os
import yaml


def main(args):
    cwd = os.path.dirname(os.path.abspath(__file__))

    in_dir = os.path.join(cwd, 'config')

    meta_file = os.path.join(in_dir, '{}.yaml'.format(args.experiment))

    pipeline_dir = os.path.join(cwd, 'pipeline')

    out_dir = os.path.join(pipeline_dir, 'config')

    out_file = os.path.join(out_dir, '{}.yaml'.format(args.experiment))

    runs_dir = os.path.join(out_dir, 'runs', args.experiment)

    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    with open(meta_file, 'r') as fh:
        meta_config = yaml.load(fh, Loader=yaml.FullLoader)

    out_config = {'runs': {}}

    for run in meta_config:
        run_config = {}

        data_set_file = os.path.join(
            in_dir, 'data_sets', '{}.yaml'.format(meta_config[run]['data_set'])
        )

        with open(data_set_file, 'r') as fh:
            run_config.update(yaml.load(fh, Loader=yaml.FullLoader))

        sampler_set_file = os.path.join(
            in_dir, 'sampler_sets', '{}.yaml'.format(meta_config[run]['sampler_set'])
        )

        with open(sampler_set_file, 'r') as fh:
            sampler_set_config = yaml.load(fh, Loader=yaml.FullLoader)

        run_config['sampler'] = load_sampler_set(sampler_set_config)

        run_out_file = os.path.join(runs_dir, '{}.yaml'.format(run))

        with open(run_out_file, 'w') as fh:
            yaml.dump(run_config, fh, default_flow_style=False)

        out_config['runs'][run] = {'config_file': os.path.relpath(run_out_file, pipeline_dir)}

    with open(out_file, 'w') as fh:
        yaml.dump(out_config, fh, default_flow_style=False)


def load_sampler_set(config):
    sampler_configs = config['sampler']

    if 'sampler_contrast' in config:
        sampler_configs.update(load_contrast(config))

    return sampler_configs


def load_contrast(config):
    shared_config = config.get('sampler_shared', {})

    sampler_configs = {}

    for sampler_type in config['sampler_contrast']:
        for sampler_param in config['sampler_contrast'][sampler_type]:
            for value in config['sampler_contrast'][sampler_type][sampler_param]:
                sampler_id = '{0}-{1}-{2}'.format(sampler_type, sampler_param, value)

                sampler_config = {}

                sampler_config['updater'] = sampler_type

                sampler_config['kwargs'] = shared_config[sampler_type].copy()

                sampler_config['kwargs'][sampler_param] = value

                sampler_configs[sampler_id] = sampler_config

    return sampler_configs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiment', required=True)

    cli_args = parser.parse_args()

    main(cli_args)
