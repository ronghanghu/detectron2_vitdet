#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sqlite3
import subprocess
import sys


"""
A script to get working dir (gluster or manifold) of a fry job.
It also caches results locally.

It prints only one line, the dir, to stdout.

Usage:
    ./dev/fb/get_working_dir.py [jobid]

"""


def get_wd_fry_local(jobid):
    process = subprocess.Popen(
        "mlcli job -r {} workdir".format(jobid),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        shell=True,
    )
    process.wait()
    res = process.stdout.read().strip().decode("utf-8")
    if process.returncode != 0:
        print(res)
        print("mlcli Failed with {}".format(process.returncode))
        sys.exit(process.returncode)
    if not res.startswith("/"):
        return None
    return res


def get_flow_job_input_params(jobid):
    fbsource = os.path.expanduser("~/fbsource")
    is_devserver = os.path.isdir(fbsource)
    if is_devserver:
        # this version needs to be able to find and run flow-cli command
        assert os.path.exists(
            "/usr/local/bin/flow-cli"
        ), "Run `sudo feature install fblearner_flow` first!"
        process = subprocess.Popen(
            "cd {} && flow-cli print-params {}".format(fbsource, jobid),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=True,
        )
        output = process.stdout.read().decode("utf-8")
        jsonobj = json.loads(output)[str(jobid)]
        return jsonobj
    else:
        # this version needs to be built with fbcode
        from fblearner.flow.external_api import FlowSession

        sess = FlowSession()
        res = sess.get_workflow_run_inputs_json(int(jobid))
        jsonobj = json.loads(res)
        return jsonobj


def get_wd_flow_params(jobid):
    """
    Get directory by looking at the flow output json of the job.
    """
    jsonobj = get_flow_job_input_params(jobid)

    if "job_gluster_home" in jsonobj:
        # check fry job params - alternative to the mlcli command
        ret = jsonobj["job_gluster_home"]
        if ret.endswith("output"):
            ret = ret[: -len("output")]
        return ret
    elif "params" in jsonobj:
        # flow job
        ret = jsonobj["params"]["output_dir"]
        if ret is None:  # recurring job, no dir is given as job param
            process = subprocess.Popen(
                "bento console -c 'from fblearner.flow.external_api import FlowSession as S;"
                "print(S().get_workflow_run_results({}).output_dir)'".format(jobid),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                shell=True,
            )
            ret = process.stdout.read().decode("utf-8").strip()
        else:
            if ret.endswith("output"):
                ret = ret[: -len("output")]
        return ret
    elif "streams" in jsonobj:
        # fry manifold job
        streams = jsonobj["streams"]
        assert len(streams) == 1
        ret = list(streams.keys())[0]
        assert ret.startswith("manifold://")
        return ret
    elif "output_dir" in jsonobj:
        return jsonobj["output_dir"]


def get_wd(jobid):
    if isinstance(jobid, str):
        jobid = int(jobid[1:])
    for e in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(e, None)
    wd = get_wd_flow_params(jobid)
    if wd:
        return wd
    # mlcli no longer works
    # wd = get_wd_fry_local(jobid)
    # if wd:
    #     return wd
    raise RuntimeError("Cannot obtain dir for {}".format(jobid))
