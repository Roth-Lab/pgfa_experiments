import glob
import h5py
import numpy as np
import os
import skimage.io
import skimage.transform
import skimage.util


def main(args):
    data = []

    img_shape = (args.size, args.size)

    files = glob.glob(os.path.join(args.in_dir, "*"))

    for file_name in files:
        if file_name.endswith(".txt"):
            continue

        img = skimage.io.imread(file_name)

        data.append(
            skimage.util.img_as_ubyte(
                skimage.transform.resize(img, img_shape, anti_aliasing=True).flatten()
            )
        )

    data = np.array(data)

    with h5py.File(args.out_file, "w") as fh:
        fh.create_dataset("data", compression="gzip", data=data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--in-dir", required=True)

    parser.add_argument("-o", "--out_file", required=True)

    parser.add_argument("-s", "--size", default=48, type=int)

    cli_args = parser.parse_args()

    main(cli_args)
