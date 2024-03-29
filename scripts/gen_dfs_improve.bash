#!/bin/bash

## Data version
data_version=July2020

## Path to raw data
# rawdir=IMP_data/raw/$data_version
rawdir=raw/$data_version

## Y data (path drug response)
# rsp_path=$rawdir/combined_single_response_rescaled_agg
rsp_path=$rawdir/combined_single_response_rescaled_agg.parquet

## X data (path cancer features)
# cell_path=$rawdir/lincs1000/combined_rnaseq_data_lincs1000
# cell_path=$rawdir/combined_rnaseq_data
cell_path=$rawdir/combined_rnaseq_data.parquet

## X data (path drug features)
# drug_path=$rawdir/drug_info/dd.mordred.with.nans
# drug_path=$rawdir/combined_drug_descriptors_dragon7_2k
# drug_path=$rawdir/combined_drug_descriptors_mordred_2k
drug_path=$rawdir/dd.mordred.csv

## Experiment settings
dropna_th=0.1
r2fit_th=0.3
# No of samples
n_samples=200  # Smaller dataset (sample n samples)
# n_samples=''  # Smaller dataset (sample n samples)

## Outdir path
if [ -z "$n_samples" ]
then
    echo "n_samples is $n_samples"
    outdir=$rawdir/../../out_data/$data_version
else
    echo "n_samples is $n_samples"
    lc_min_size=5
    echo "lc_min_size is $lc_min_size"
    outdir=$rawdir/../../out_data_sample/$data_version
fi
# outdir=$rawdir/../../ml.dfs/$data_version
# outdir=$rawdir/../../ml.dfs.tr_vl_te/$data_version
# -----
# Use this ..
# outdir=$rawdir/../../out_data/$data_version
# .. or this
# outdir=$rawdir/../../out_data_sample/$data_version
# -----

# ------------------------------------------------------------
sources=("ccle" "ctrp" "gcsi" "gdsc1" "gdsc2")
# sources=("gdsc1" "gdsc2")
# sources=("ccle" "gdsc2")
# sources=("gdsc1")
# sources=("gdsc2")
# sources=("ccle")

for SOURCE in ${sources[@]}; do
    # python IMP_data/src/build_dfs_july2020.py \
    python src/build_dfs_july2020.py \
        --src $SOURCE \
        --rsp_path $rsp_path \
        --cell_path $cell_path \
        --drug_path $drug_path \
        --dropna_th $dropna_th \
        --r2fit_th $r2fit_th \
        --n_samples $n_samples \
        --lc_min_size $lc_min_size \
        --gout $outdir
done

# ------------------------------------------------------------
# drug_path=$rawdir/NCI60_drugs_52k_smiles/dd.mordred.with.nans

# SOURCE='nci60'
# python src/build_dfs_july2020.py \
#     --src $SOURCE \
#     --rsp_path $rsp_path \
#     --cell_path $cell_path \
#     --drug_path $drug_path \
#     --dropna_th $dropna_th \
#     --r2fit_th $r2fit_th \
#     --n_samples 750000 \
#     --flatten \
#     --gout $outdir
