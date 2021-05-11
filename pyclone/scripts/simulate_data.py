import numpy as np
import pandas as pd
import scipy.stats
import yaml

import pgfa.models.pyclone.binomial


def main(args):
    with open(args.config_file, 'r') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)['data']

    if args.seed is not None:
        np.random.seed(args.seed)

    params = pgfa.models.pyclone.binomial.simulate_params(
        D=config['D'], N=config['N'], K=config['K'], alpha=config['alpha']
    )

    data = simulate_data(params, eps=1e-3)

    with pd.HDFStore(args.out_file, 'w') as store:
        store['data'] = data

        store['alpha'] = pd.Series([params.alpha])

        store['V'] = pd.DataFrame(params.V)

        store['Z'] = pd.DataFrame(params.Z)

def simulate_data(params, eps=1e-3):
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

            xi = (1 - t[s]) * phi[s] * cn_n * mu_n + t[s] * (1 - phi[s]) * cn_r * mu_r + t[s] * phi[s] * cn_total * mu_v

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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config-file', required=True)

    parser.add_argument('-o', '--out-file', required=True)

    parser.add_argument('--seed', default=None, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
