#!/bin/bash


# data_version=July2020
# rawdir=IMP_data/raw/$data_version
# outdir=$rawdir/../../ml.dfs/$data_version

# rsp_path=$rawdir/combined_single_response_rescaled_agg.parquet

# cell_path=$rawdir/combined_rnaseq_data.parquet



SOURCE=$1
data_version=July2020

dfdir=IMP_data/ml.dfs/$data_version

dpath=$dfdir/data.$SOURCE/data.$SOURCE.mordred.ge.parquet
gout=$dfdir/data.$SOURCE

# sampling=random
# sampling=flatten

# dpath=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$sampling/data.$SOURCE.dd.ge.parquet
# gout=data/ml.dfs/$data_version/data.$SOURCE.dd.ge.$sampling

python IMP_data/src/main_data_split.py \
    -dp $dpath \
    --gout $gout \
    --trg_name AUC_bin \
    -ns 10 \
    -cvm strat \
    --te_size 0.10
