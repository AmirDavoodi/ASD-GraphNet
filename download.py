import pandas as pd
import os
from tqdm import tqdm
import wget


# base_str = https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/[pipeline]/[strategy]/[derivative]/[file identifier]_[derivative].[ext]


# https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_aal/KKI_0050822_rois_aal.1D
base_str = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/[pipeline]/[filt]/[roi]/[file identifier]_[roi].1D"


roi_str = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/[pipeline]/filt_global/[derivative]/[file identifier]_[derivative].1D"


def create_url(pipe, roi, fg, file_id):
    url = base_str.replace("[pipeline]", pipe)
    url = url.replace("[filt]", fg)
    url = url.replace("[roi]", roi)
    url = url.replace("[file identifier]", file_id)
    return url


def download_abide1_roi(pheno_file, out_dir, pipe, roi, fg):
    df = pd.read_csv(pheno_file)

    # create out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row["FILE_ID"]

        if file_id == "no_filename":
            continue
        url = create_url(pipe, roi, fg, file_id)
        out_file = f"{out_dir}/{file_id}_{roi}.1D"
        try:
            print(f"Downloading {url} to {out_file}")
            wget.download(url, out_file, bar=None)
            print(f"\nDownloaded {url} to {out_file}")
        except Exception as e:
            print(e)
            print(f"Failed to download {url} to {out_file}")


if __name__ == "__main__":
    # argument parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pheno_file", type=str)
    parser.add_argument("out_dir", type=str)
    # parser.add_argument("--pipe", type=str, default="dparsf")
    parser.add_argument("--pipe", type=str, default="cpac")
    parser.add_argument("--roi", type=str, default="rois_cc200")
    parser.add_argument("--fg", type=str, default="filt_global")
    args = parser.parse_args()

    download_abide1_roi(
        args.pheno_file,
        args.out_dir,
        args.pipe,
        args.roi,
        args.fg)
