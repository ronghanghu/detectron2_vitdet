#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import pathlib
import pickle
import shutil
import torch

try:
    from iopath.common.file_io import PathManager
    from iopath.fb.manifold import ManifoldPathHandler

    PathManager.register_handler(ManifoldPathHandler())
except ImportError:
    pass

"""
Example:

faircluster:
    ./convert_to_model_zoo.py --cluster fair \
    --job-dir /checkpoint/bowencheng/experiments/30841561/output --output \
    PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32

devgpu:
    bento console --kernel vision.detectron2 --file convert_to_model_zoo.py
      -- -- --job-dir manifold://xx/output --cluster fb --output abcdefg
"""

DEFAULT_CLOUD_ROOT = {
    "fair": "s3://dl.fbaipublicfiles.com/detectron2",
    "fb": "fair_vision_data/tree/detectron2/model_zoo",
}
DEFAULT_LOCAL_ROOT = {
    "fair": "/private/home/yuxinwu/data/D2models",  # on H2
    "fb": "/mnt/vol/gfsai-bistro-east/ai-group/users/vision/detectron2",
}


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
    if model_file.startswith("manifold://"):
        model_file = PathManager.open(model_file, "rb")

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
    parser.add_argument("--cluster", required=True, choices=["fb", "fair"])
    parser.add_argument(
        "--output", required=True, help="output dir name relative to the model zoo root"
    )
    parser.add_argument(
        "--local-root",
        help="path to a local directory of model zoo; Default to detectron2's location",
    )
    parser.add_argument(
        "--cloud-root",
        help="path to the cloud storage of model zoo; "
        "Default to detectron2's mainfold or s3 bucket.",
    )
    args = parser.parse_args()

    if args.local_root is None:
        args.local_root = DEFAULT_LOCAL_ROOT[args.cluster]
    if args.cloud_root is None:
        args.cloud_root = DEFAULT_CLOUD_ROOT[args.cluster]
    if args.cluster == "fb":
        assert ManifoldPathHandler is not None, "run inside bento console!"
    assert os.path.isdir(args.local_root)

    output_dir = pathlib.Path(args.local_root) / args.output
    os.makedirs(output_dir, exist_ok=True)

    model = get_model(os.path.join(args.job_dir, "model_final.pth"))

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
        source = os.path.join(args.job_dir, f)
        if source.startswith("manifold://"):
            source = PathManager.get_local_path(source)
        shutil.copy(source, output_dir / f)

    if args.cluster == "fair":
        message = f"""
    Please run the following (on FAIR cluster):

        module load fairusers_aws
        fs3cmd sync -F {args.local_root}/ {args.cloud_root}/ --exclude '*.lock' --exclude 'log.txt' --exclude '.git*' -n

    If results look OK, remove "-n" to launch the actual sync.
    """  # noqa
    else:
        local_output_dir = str(output_dir).rstrip("/")
        rel_output_dir = args.output.rstrip("/")
        message = f"""
    Please run the following (on FB devserver):

        manifold mkdirs {args.cloud_root}/{rel_output_dir}
        manifold putr {output_dir} {args.cloud_root}/{rel_output_dir}
    """
    print(message)
