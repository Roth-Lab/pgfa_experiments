import numpy as np
import pandas as pd


def main(args):
    np.random.seed(args.seed)

    clone_df = load_clone_file(args.clone_file)

    cell_to_samples_df = get_cell_to_samples(clone_df, args.num_samples)

    ccf_df = get_ccf_df(cell_to_samples_df, clone_df)

    cnv_df = load_cnv_file(cell_to_samples_df, args.cn_file)

    snv_df = load_snv_file(cell_to_samples_df, args.snv_file)

    # Remove SNVs not present in all samples
    snv_df = snv_df.groupby("mutation_id").filter(lambda x: x.shape[0] == args.num_samples)

    data_df = get_data_df(cnv_df, snv_df)

    clone_snv_df = load_clone_snv_file(args.clone_snv_file)

    data_df = data_df[data_df["mutation_id"].isin(clone_snv_df.index)]

    data_df = data_df.sort_values(by=["mutation_id", "sample_id"])

    ccf_df.to_csv(args.out_ccf_file, compression="gzip", index=False, sep="\t")

    data_df.to_csv(args.out_data_file, compression="gzip", index=False, sep="\t")

    clone_snv_df.to_csv(args.out_ground_truth_file, compression="gzip", sep="\t")


def get_ccf_df(cells_to_samples, clone_df):
    clone_sample_map = pd.merge(
        cells_to_samples,
        clone_df,
        on="cell_id"
    )

    total_cells = clone_sample_map.groupby("sample_id").apply(lambda x: x.shape[0])

    total_cells = total_cells.reset_index()

    total_cells.columns = "sample_id", "total_cells"

    clone_cells = clone_sample_map.groupby(["sample_id", "clone_id"]).apply(lambda x: x.shape[0])

    clone_cells = clone_cells.reset_index()

    clone_cells.columns = "sample_id", "clone_id", "clone_cells"

    ccf = pd.merge(total_cells.reset_index(), clone_cells.reset_index(), on="sample_id")

    ccf["ccf"] = ccf["clone_cells"] / ccf["total_cells"]

    ccf = ccf[["sample_id", "clone_id", "ccf"]]

    return ccf


def get_cell_to_samples(clone_df, num_samples):
    """ Randomly assign cells to fake samples.
    """
    cells = list(clone_df["cell_id"].unique())

    cells_to_samples = []

    while len(cells) > 0:
        sample_idx = np.random.randint(num_samples)

        cell_idx = np.random.randint(len(cells))

        cells_to_samples.append({
            "cell_id": cells.pop(cell_idx),
            "sample_id": sample_idx
        })

    return pd.DataFrame(cells_to_samples)


def get_data_df(cnv_df, snv_df):
    data_df = []
    for sample_id in snv_df["sample_id"].unique():
        sample_cnv_df = cnv_df[cnv_df["sample_id"] == sample_id]

        sample_snv_df = snv_df[snv_df["sample_id"] == sample_id]

        for _, row in sample_snv_df.iterrows():
            chrom, coord, _, _ = row["mutation_id"].split(":")

            coord = int(coord)

            x = sample_cnv_df[
                (sample_cnv_df["chrom"] == chrom) & (sample_cnv_df["beg"] <= coord) & (sample_cnv_df["end"] >= coord)
            ]["cn"]

            if x.empty:
                continue

            cn = x.values[0]

            out_row = {
                "sample_id": sample_id,
                "mutation_id": row["mutation_id"],
                "ref_counts": row["ref_counts"],
                "alt_counts": row["alt_counts"],
                "cn": int(cn)
            }

            data_df.append(out_row)

    return pd.DataFrame(data_df)


def load_clone_file(file_name):
    return pd.read_csv(file_name)


def load_clone_snv_file(file_name):
    clone_snv_df = pd.read_csv(file_name)

    clone_snv_df["mutation_id"] = clone_snv_df.apply(
        lambda x: ":".join([str(x["chrom"]), str(x["coord"]), x["ref"], x["alt"]]), axis=1
    )

    clone_snv_df = clone_snv_df.groupby("mutation_id").filter(lambda x: x["is_present"].sum() > 0)

    clone_snv_df = clone_snv_df.pivot(index="mutation_id", columns="clone_id", values="is_present")

    return clone_snv_df


def load_cnv_file(cells_to_samples, file_name):
    cnv_df = pd.read_csv(file_name)

    # Remove original sample IDs
    cnv_df = cnv_df.drop("sample_id", axis=1)

    # Add new sample IDs and filter for cells with clone annotations
    cnv_df = pd.merge(cnv_df, cells_to_samples, on="cell_id", how="inner")

    # Derive sample level CNVs
    cnv_df = cnv_df.groupby(["sample_id", "chr", "start", "end"]).apply(lambda x: int(x["state"].median()))

    cnv_df = cnv_df.reset_index()

    cnv_df.columns = "sample_id", "chrom", "beg", "end", "cn"

    return cnv_df


def load_snv_file(cells_to_samples, file_name):
    snv_df = pd.read_csv(file_name)

    # Remove original sample IDs
    snv_df = snv_df.drop("sample_id", axis=1)

    # Add new sample IDs and filter for cells with clone annotations
    snv_df = pd.merge(snv_df, cells_to_samples, on="cell_id", how="inner")

    # Add unique mutation ID
    snv_df["mutation_id"] = snv_df.apply(lambda x: ":".join(
        [str(x["chrom"]), str(x["coord"]), x["ref"], x["alt"]]), axis=1)

    # Get sample level SNV allele counts
    snv_df = snv_df.groupby(["sample_id", "mutation_id"]).apply(lambda x: x[["ref_counts", "alt_counts"]].sum())

    snv_df = snv_df.reset_index()

    return snv_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--clone-file", required=True)

    parser.add_argument("--clone-snv-file", required=True)

    parser.add_argument("--cn-file", required=True)

    parser.add_argument("--snv-file", required=True)

    parser.add_argument("--out-ccf-file", required=True)

    parser.add_argument("--out-data-file", required=True)

    parser.add_argument("--out-ground-truth-file", required=True)

    parser.add_argument("--num-samples", required=True, type=int)

    parser.add_argument("--seed", required=True, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
