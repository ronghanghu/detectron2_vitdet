#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import pathlib
import pickle
import shutil
import torch

"""
Example:
    ./convert_to_model_zoo.py \
    --job-dir /checkpoint/bowencheng/experiments/30841561/output --output \
    PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32
"""

S3_ROOT = "s3://dl.fbaipublicfiles.com/detectron2/"
DEFAULT_CLUSTER_ROOT = "/private/home/yuxinwu/data/D2models"  # on H2


def get_md5(file_name):
    with open(file_name, "rb") as f:
        s = f.read()
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()


def get_model(model_file):
    """
    Read a checkpoint, clean up unnecessary states (optimizer, etc)
    """
    obj = torch.load(model_file)
    for k in list(obj.keys()):
        if k != "model":
            del obj[k]
    obj["__author__"] = "Detectron2 Model Zoo"

    model = obj["model"]
    for k in list(model.keys()):
        model[k] = model[k].cpu().numpy()
    # model is an OrderedDict, that contains an extra "_metadata" attribute
    # We modify it in-place so that these properties stay untouched
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True)
    parser.add_argument(
        "--output", required=True, help="output dir name relative to the model zoo root"
    )
    parser.add_argument("--zoo-root", default=DEFAULT_CLUSTER_ROOT)
    args = parser.parse_args()
    assert os.path.isdir(args.zoo_root)

    output_dir = pathlib.Path(args.zoo_root) / args.output
    jobdir = pathlib.Path(args.job_dir)
    os.makedirs(output_dir, exist_ok=True)

    model = get_model(jobdir / "model_final.pth")

    tmp_output_model = output_dir / "model_final.pkl"
    print(f"Writing models to {tmp_output_model} ...")
    with open(tmp_output_model, "wb") as f:
        pickle.dump(model, f)

    md5 = get_md5(tmp_output_model)[:6]
    output_model = output_dir / f"model_final_{md5}.pkl"
    print(f"Renaming to {output_model} ...")
    shutil.move(tmp_output_model, output_model)

    for f in ["log.txt", "metrics.json"]:
        print(f"Copying {f} ...")
        shutil.copy(jobdir / f, output_dir / f)

    message = f"""
Please run the following (on FAIR cluster):

    module load fairusers_aws
    fs3cmd sync -F {args.zoo_root}/ {S3_ROOT} --exclude '*.lock' --exclude 'log.txt' -n

If results look OK, remove "-n" to launch the actual sync.
"""
    print(message)
