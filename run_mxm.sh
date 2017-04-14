#!/usr/bin/env bash
export PYTHONPATH=${pwd}:PYTHONPATH
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=4
mkdir -p mxm/log/runtime
mkdir -p mxm/log/train
python mxm/release/res_net_50/train.py > mxm/log/runtime/mxnet.log 2>&1