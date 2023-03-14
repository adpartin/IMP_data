#!/bin/bash --login

set -e

# This script was create with the help from here:
# https://www.youtube.com/watch?v=lu2DzaqBeDg
# https://github.com/kaust-rccl/conda-environment-examples/tree/pytorch-geometric-and-friends


# conda create -n imp_data python=3.7 pip --yes

conda install -c conda-forge mordred=1.2 --yes
conda install -c conda-forge scikit-learn --yes
conda install -c conda-forge seaborn --yes
conda install -c conda-forge psutil --yes

# conda install -c conda-forge numpy=1.24 --yes # installed with mordred
# conda install -c conda-forge pandas=1.5 --yes # installed with mordred
# conda install -c conda-forge matplotlib=3.7 --yes # installed with mordred

conda install -c bioconda pubchempy=1.0.4 --yes
# conda install -c rdkit rdkit --yes # installed with mordred
# conda install -c anaconda networkx --yes

conda install -c conda-forge pyarrow --yes
conda install -c conda-forge h5py --yes

pip install lightgbm

# CANDLE
# pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
# pip install git+https://github.com/ECP-CANDLE/candle_lib@candle_data_dir

# My packages
conda install -c conda-forge ipdb=0.13.11 --yes
conda install -c conda-forge python-lsp-server=1.2.4 --yes
