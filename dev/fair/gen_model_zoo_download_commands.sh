#!/bin/bash -e

# This scripts runs on faircluster. It generate wget-based download commands
# that can download a directory from s3 to other machines (e.g. a devserver)
# Example:
# ./gen_download_command.sh s3://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/

module load fairusers_aws
path=$1
fs3cmd ls -r "$path" | awk '{print $4}' | while read -r file; do
  url=${file//s3/https}
  echo "wget $url -x -nH --cut-dirs 1"
done
