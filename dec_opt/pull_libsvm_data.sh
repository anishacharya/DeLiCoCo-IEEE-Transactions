#!/usr/bin/env bash

# Pulls and creates pickle files for libsvm data

curr_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
data_dir=$curr_dir'../data'

mkdir -p "$data_dir"

datasets=${1-epsilon}

if [ "$datasets" == 'epsilon' ] || [ "$datasets" == 'all' ];
then
	if [ -f "$data_dir"'/epsilon_normalized.pickle' ]; then echo "epsilon exists skipping download"
	else
		wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
		mv 'epsilon_normalized.bz2' "$data_dir"
		echo "Processing epsilon"
		python3 get_data_helper.py --i "$data_dir"'/epsilon_normalized.bz2' --o "$data_dir"'/epsilon_normalized.pickle'
	fi
fi

if [ "$datasets" == 'rcv1' ] || [ "$datasets" == 'all' ];
then
	if [ -f "$data_dir"'/rcv1_test.binary.pickle' ]; then echo "rcv1 exists skipping download"
	else
		wget -t inf https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
		mv 'rcv1_test.binary.bz2' "$data_dir"
		echo "Processing rcv1"
		python3 get_data_helper.py --i "$data_dir"'/rcv1_test.binary.bz2' --o "$data_dir"'/rcv1_test.pickle'
	fi
fi




