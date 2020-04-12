#!/usr/bin/env bash
curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
data_dir=$curr_dir'/data'

mkdir -p "$data_dir"

wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2

mv 'epsilon_normalized.bz2' "$data_dir"
mv 'rcv1_test.binary.bz2' "$data_dir"

python pickle_datasets.py