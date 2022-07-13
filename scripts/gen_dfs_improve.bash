#!/bin/bash

data_version=July2020
rawdir=IMP_data/raw/$data_version
# outdir=$rawdir/../../ml.dfs/$data_version
outdir=$rawdir/../../ml.dfs/$data_version

# rsp_path=$rawdir/combined_single_response_rescaled_agg
rsp_path=$rawdir/combined_single_response_rescaled_agg.parquet

# cell_path=$rawdir/lincs1000/combined_rnaseq_data_lincs1000
# cell_path=$rawdir/combined_rnaseq_data
cell_path=$rawdir/combined_rnaseq_data.parquet

dropna_th=0.1
r2fit_th=0.3

# ------------------------------------------------------------
sources=("ccle" "ctrp" "gcsi" "gdsc1" "gdsc2")
# sources=("gdsc1" "gdsc2")
# sources=("gdsc1")

# drug_path=$rawdir/drug_info/dd.mordred.with.nans
# drug_path=$rawdir/combined_drug_descriptors_dragon7_2k
# drug_path=$rawdir/combined_drug_descriptors_mordred_2k
drug_path=$rawdir/dd.mordred.csv

for SOURCE in ${sources[@]}; do
    # python src/build_dfs_july2020.py \
    python IMP_data/src/build_dfs_july2020.py \
        --src $SOURCE \
        --rsp_path $rsp_path \
        --cell_path $cell_path \
        --drug_path $drug_path \
        --dropna_th $dropna_th \
        --r2fit_th $r2fit_th \
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
