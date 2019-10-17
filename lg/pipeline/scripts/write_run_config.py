import os
import yaml


def main(args):
    pass


def write_config_files(in_file, out_files):
    with open(in_file, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    for sampler_set in config['sampler_sets']:
        sampler_set_file = os.path.join(CONFIG_DIR, 'sampler_sets', '{}.yaml'.format(sampler_set))

        with open(sampler_set_file, 'r') as fh:
            sampler_set_config = yaml.load(fh, Loader=yaml.FullLoader)

        experiment_config = config['experiment'].copy()

        experiment_config['sampler'] = _load_sampler_set(sampler_set_config)

        out_file = out_files[sampler_set]

        with open(out_file, 'w') as fh:
            yaml.dump(experiment_config, fh, default_flow_style=False)


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

                sampler_config = shared_config[sampler_type].copy()

                sampler_config['updater'] = sampler_type

                sampler_config[sampler_param] = value

                sampler_configs[sampler_id] = sampler_config

    return sampler_configs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--in-file', required=True)

    parser.add_argument('-o', '--out-file', required=True)

    parser.add_argument('-r', '--run-id', required=True)

    cli_args = parser.parse_args()

    main(cli_args)
