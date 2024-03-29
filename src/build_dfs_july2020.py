"""
This "new" version of the code uses a different dataframe for descriptors:
'pan_drugs_dragon7_descriptors.tsv' instead of 'combined_pubchem_dragon7_descriptors.tsv'
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
import json
from time import time
from pprint import pformat

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer

# github.com/mtg/sms-tools/issues/36
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
from utils.classlogger import Logger
from utils.utils import dump_dict, get_print_func, dropna
from utils.impute import impute_values
from utils.resample import flatten_dist
from ml.scale import scale_fea
from ml.data import extract_subset_fea

# File path
filepath = Path(__file__).resolve().parent

# Settings
metadata = {}
na_values = ['na', '-', '']
fea_prfx_dct = {'ge': 'ge_', 'cnv': 'cnv_', 'mu': 'mu_', 'snp': 'snp_',
                'mordred': 'mordred_', 'dragon': 'dragon_', 'fng': 'fng_'}
# TODO: consider to make this types SimpleNamspace
canc_col_name = "CancID"
drug_col_name = "DrugID"
metadata["CANC_COL_NAME"] = canc_col_name
metadata["DRUG_COL_NAME"] = drug_col_name
x_datadir_name = "x_data"
y_datadir_name = "y_data"
splits_datadir_name = "splits"
splits_sorted_datadir_name = "splits_sorted"
lc_splits_datadir_name = "lc_splits"
misc_datadir_name = "misc_data"
metadata["X_DATADIR_NAME"] = x_datadir_name
metadata["Y_DATADIR_NAME"] = y_datadir_name
metadata["SPLITS_DATADIR_NAME"] = splits_datadir_name
metadata["SPLITS_SORTED_DATADIR_NAME"] = splits_sorted_datadir_name
metadata["LC_SPLITS_DIRNAME"] = lc_splits_datadir_name
metadata["MISC_DATADIR_NAME"] = misc_datadir_name

# Rename columns that contain cancer IDs and drug IDs
rename_id_col_names = {"CELL": canc_col_name, "DRUG": drug_col_name}


def create_basename(args):
    """ Name to characterize the data. Can be used for dir name and file name. """
    # ls = args.drug_fea + args.cell_fea
    # if args.src is None:
    #     name = '.'.join(ls)
    # else:
    #     src_names = '_'.join(args.src)
    #     name = '.'.join([src_names] + ls)
    name = '_'.join(args.src)  # isntead of the code above

    name = 'data.' + name
    return name


def create_outdir(outdir, args):
    """ Creates output dir. """
    basename = create_basename(args)
    outdir = Path(outdir, basename)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def groupby_src_and_print(df, col1="CELL", col2="DRUG", print_fn=print):
    print_fn(df.groupby('SOURCE').agg({col1: 'nunique', col2: 'nunique'}).reset_index())


def add_fea_prfx(df, prfx: str, id0: int):
    """ Add prefix feature columns. """
    return df.rename(columns={s: prfx+str(s) for s in df.columns[id0:]})


def read_df(fpath, sep="\t"):
    assert Path(fpath).exists(), f"File {fpath} was not found."
    if "parquet" in str(fpath):
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath, sep=sep, na_values=na_values)
    return df


def load_rsp(fpath, src=None, r2fit_th=None, print_fn=print):
    """ Load drug response data.
    src : extract specific data sources (i.e., studies)
    r2fit_th : threshold used to filter out samples with low r2 of the dose-response curve fit
    """
    rsp = read_df(fpath)
    if "STUDY" in rsp.columns:
        rsp.drop(columns="STUDY", inplace=True)  # gives error when saves in 'parquet' format
    # print(rsp.dtypes)

    print_fn('\nAll samples (original).')
    print_fn(f'rsp.shape {rsp.shape}')
    groupby_src_and_print(rsp, print_fn=print_fn)
    print_fn(rsp.SOURCE.value_counts())

    # Drop bad samples
    if r2fit_th is not None:
        # Yitan
        # TODO: check this (may require a more rigorous filtering)
        # print_fn('\n\nDrop bad samples ...')
        # id_drop = (rsp['AUC'] == 0) & (rsp['EC50se'] == 0) & (rsp['R2fit'] == 0)
        # rsp = rsp.loc[~id_drop,:]
        # print_fn(f'Dropped {sum(id_drop)} rsp data points.')
        # print_fn(f'rsp.shape {rsp.shape}')
        print_fn('\nDrop samples with low R2fit.')
        print_fn('Samples with bad fit.')
        id_drop = rsp['R2fit'] <= r2fit_th
        rsp_bad_fit = rsp.loc[id_drop, :].reset_index(drop=True)
        groupby_src_and_print(rsp_bad_fit, print_fn=print_fn)
        print_fn(rsp_bad_fit.SOURCE.value_counts())

        print_fn('\nSamples with good fit.')
        rsp = rsp.loc[~id_drop, :].reset_index(drop=True)
        groupby_src_and_print(rsp, print_fn=print_fn)
        print_fn(rsp.SOURCE.value_counts())
        print_fn(f'Dropped {sum(id_drop)} rsp data points.')

    rsp['SOURCE'] = rsp['SOURCE'].apply(lambda x: x.lower())

    if src is not None:
        print_fn('\nExtract specific sources.')
        rsp = rsp[rsp['SOURCE'].isin(src)].reset_index(drop=True)

    # Create a binary label from AUC
    # TODO: there are better ways. E.g., per-drug thersholds)
    rsp['AUC_bin'] = rsp['AUC'].map(lambda x: 0 if x > 0.5 else 1)
    rsp.replace([np.inf, -np.inf], value=np.nan, inplace=True)

    print_fn(f'rsp.shape {rsp.shape}')

    groupby_src_and_print(rsp, print_fn=print_fn)
    return rsp


def load_ge(fpath, impute=True, print_fn=print, float_type=np.float32):
    """ Load RNA-Seq data. """
    print_fn(f'\nLoad RNA-Seq ... {fpath}')
    # ge = pd.read_csv(fpath, sep='\t', na_values=na_values)
    ge = read_df(fpath)
    ge.rename(columns={'Sample': 'CELL'}, inplace=True)

    fea_id0 = 1
    ge = add_fea_prfx(ge, prfx=fea_prfx_dct['ge'], id0=fea_id0)

    if sum(ge.isna().sum() > 0) and impute:
        # ge = impute_values(ge, print_fn=print_fn)

        print_fn('Columns with NaNs: {}'.format( sum(ge.iloc[:, fea_id0:].isna().sum() > 0) ))
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imputer = KNNImputer(missing_values=np.nan, n_neighbors=5,
        #                      weights='uniform', metric='nan_euclidean',
        #                      add_indicator=False)
        ge.iloc[:, fea_id0:] = imputer.fit_transform(ge.iloc[:, fea_id0:].values)
        print_fn('Columns with NaNs: {}'.format( sum(ge.iloc[:, fea_id0:].isna().sum() > 0) ))

    # Cast features (casting to float16 changes the shape. why?)
    ge = ge.astype(dtype={c: float_type for c in ge.columns[fea_id0:]})
    print_fn(f'ge.shape {ge.shape}')
    return ge


def load_dd(fpath, impute=True, print_fn=print, dropna_th=0.1, float_type=np.float32, src=None):
    """ Load drug descriptors. """
    print_fn(f'\nLoad descriptors ... {fpath}')
    # dd = pd.read_csv(fpath, sep='\t', na_values=na_values)
    dd = read_df(fpath)
    dd.rename(columns={'ID': 'DRUG'}, inplace=True)

    # dd = add_fea_prfx(dd, prfx=fea_prfx_dct['dd'], id0=fea_id0)

    if "dragon" in fpath:
        fea_id0 = 1
    elif "nci60" in src:
        dd = dropna(dd, axis=0, th=dropna_th)
        fea_id0 = 2
    else:
        fea_id0 = 5

    if sum(dd.isna().sum() > 0) and impute:
        print_fn('Columns with all NaN values: {}'.format(
            sum(dd.isna().sum(axis=0).sort_values(ascending=False) == dd.shape[0])))
        print_fn('Columns with NaNs: {}'.format( sum(dd.iloc[:, fea_id0:].isna().sum() > 0) ))
        imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imputer = KNNImputer(missing_values=np.nan, n_neighbors=5,
        #                      weights='uniform', metric='nan_euclidean',
        #                      add_indicator=False)
        dd.iloc[:, fea_id0:] = imputer.fit_transform(dd.iloc[:, fea_id0:].values)
        print_fn('Columns with NaNs: {}'.format( sum(dd.iloc[:, fea_id0:].isna().sum() > 0) ))

    # Cast features
    dd = dd.astype(dtype={c: float_type for c in dd.columns[fea_id0:]})
    print_fn(f'dd.shape {dd.shape}')
    return dd


def plot_dd_na_dist(dd, savepath=None):
    """ Plot distbirution of na values in drug descriptors. """
    fig, ax = plt.subplots()
    sns.distplot(dd.isna().sum(axis=0)/dd.shape[0], bins=100, kde=False, hist_kws={'alpha': 0.7})
    plt.xlabel('Ratio of total NA values in a descriptor to the total drug count')
    plt.ylabel('Total # of descriptors with the specified NA ratio')
    plt.title('Histogram of descriptors based on ratio of NA values')
    plt.grid(True)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight') # dpi=200
    else:
        plt.savefig('dd_hist_ratio_of_na.png', bbox_inches='tight') # dpi=200


def plot_rsp_dists(rsp, rsp_cols, savepath=None):
    """ Plot distributions of all response variables.
    Args:
        rsp : df of response values
        rsp_cols : list of col names
        savepath : full path to save the image
    """
    ncols = 4
    nrows = int(np.ceil(len(rsp_cols)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, figsize=(10, 10))
    for i, ax in enumerate(axes.ravel()):
        if i >= len(rsp_cols):
            fig.delaxes(ax)  # delete un-used ax
        else:
            target_name = rsp_cols[i]
            x = rsp[target_name].copy()
            x = x[~x.isna()].values
            sns.distplot(x, bins=100, kde=True, ax=ax, label=target_name,  # fit=norm,
                         kde_kws={'color': 'k', 'lw': 0.4, 'alpha': 0.8},
                         hist_kws={'color': 'b', 'lw': 0.4, 'alpha': 0.5})
            ax.tick_params(axis='both', which='major', labelsize=7)
            txt = ax.yaxis.get_offset_text(); txt.set_size(7)  # adjust exponent fontsize in xticks
            txt = ax.xaxis.get_offset_text(); txt.set_size(7)
            ax.legend(fontsize=5, loc='best')
            ax.grid(True)

    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight')  # dpi=200
    else:
        plt.savefig('rsp_dists.png', bbox_inches='tight')


def parse_args(args):
    parser = argparse.ArgumentParser(description='Create ML dataframe.')

    parser.add_argument('--rsp_path',
                        type=str,
                        help='Path to drug response file.')
    parser.add_argument('--drug_path',
                        type=str,
                        help='Path to drug features file.')
    parser.add_argument('--cell_path',
                        type=str,
                        help='Path to cell features file.')
    parser.add_argument('--r2fit_th',
                        type=float,
                        default=0.5,
                        help='Drop drug response values with R-square fit \
                        less than this value (Default: 0.5).')

    parser.add_argument('--drug_fea',
                        type=str,
                        nargs='+',
                        choices=['mordred'],
                        default=['mordred'],
                        help='Default: [mordred].')
    parser.add_argument('--cell_fea',
                        type=str,
                        nargs='+',
                        choices=['ge'],
                        default=['ge'],
                        help='Default: [ge].')
    parser.add_argument('--gout',
                        type=str,
                        help='Default: ...')
    parser.add_argument('--dropna_th',
                        type=float,
                        default=0,
                        help='Default: 0')
    parser.add_argument('--src',
                        nargs='+',
                        default=None,
                        choices=['ccle', 'gcsi', 'gdsc', 'gdsc1', 'gdsc2', 'ctrp', 'nci60'],
                        help='Data sources to extract (default: None).')

    parser.add_argument('--n_samples',
                        type=int,
                        default=None,
                        help='Number of docking scores to get into the ML df (default: None).')
    parser.add_argument('--flatten',
                        action='store_true',
                        help='Flatten the distribution of response values (default: False).')
    parser.add_argument('-t', '--trg_name',
                        default='AUC',
                        type=str,
                        choices=['AUC'],
                        help='Name of target variable (default: AUC).')

    # Learning curve
    parser.add_argument('--lc_sizes',
                        type=int,
                        default=11,
                        help="Number of sizes in the learning curve (used only if the lc_step_scale is 'linear')(default: None).")
    parser.add_argument('--lc_min_size',
                        type=int,
                        default=128,
                        help="Min size value in the case when lc_step_scale is 'log2' or 'log10'")
    parser.add_argument('--lc_max_size',
                        type=int,
                        default=None,
                        help="Max size value in the case when lc_step_scale is 'log2' or 'log10'")
    parser.add_argument('--lc_step_scale',
                        type=str,
                        default="log2",
                        help="specifies how to generate the size values. Available values: 'linear', 'log2', 'log10'.")

    args = parser.parse_args(args)
    return args


def run(args):
    # import ipdb; ipdb.set_trace(context=5)
    t0 = time()
    rsp_cols = ['AUC', 'AUC1', 'EC50', 'EC50se', 'R2fit',
                'Einf', 'IC50', 'HS', 'AAC1', 'DSS1']

    # Create main outdir
    outdir = create_outdir(args.gout, args)
    # Define sub-outdirs
    y_outdir = outdir/y_datadir_name
    x_outdir = outdir/x_datadir_name
    split_outdir = outdir/splits_datadir_name
    split_sorted_outdir = outdir/splits_sorted_datadir_name
    lc_split_outdir = outdir/lc_splits_datadir_name
    misc_outdir = outdir/misc_datadir_name
    # Make outdirs
    os.makedirs(y_outdir, exist_ok=True)
    os.makedirs(x_outdir, exist_ok=True)
    os.makedirs(split_outdir, exist_ok=True)
    os.makedirs(split_sorted_outdir, exist_ok=True)
    os.makedirs(lc_split_outdir, exist_ok=True)
    os.makedirs(misc_outdir, exist_ok=True)

    # -----------------------------------------------
    #     Logger
    # -----------------------------------------------
    lg = Logger(outdir/'gen.df.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(vars(args))}')
    dump_dict(vars(args), outpath=outdir/'gen.df.args')
    json_object = json.dumps(vars(args), indent=4)
    with open(outdir/"args.json", "w") as outfile:
        outfile.write(json_object)

    # -----------------------------------------------
    #     Load response data and features
    # -----------------------------------------------
    # import ipdb; ipdb.set_trace(context=5)

    # --------
    # Y data - drug response
    # --------
    rsp = load_rsp(args.rsp_path, src=args.src, r2fit_th=args.r2fit_th,
                   print_fn=print_fn).reset_index(drop=True)
    rsp = rsp.rename(columns=rename_id_col_names)
    # Save rsp df
    # rsp_fname = "_".join(args.src)
    # rsp.to_parquet(outdir/f"rsp_full_{rsp_fname}.parquet", index=False)
    # rsp.to_csv(outdir/f"rsp_full_{rsp_fname}.csv", index=False)
    rsp_fname = "rsp_full"
    rsp.to_parquet(y_outdir/f"{rsp_fname}.parquet", index=False)
    rsp.to_csv(y_outdir/f"{rsp_fname}.csv", index=False)

    # ---------------
    # X data - Gene expression
    # ---------------
    # import ipdb; ipdb.set_trace(context=5)
    ge = load_ge(args.cell_path, impute=False, print_fn=print_fn, float_type=np.float32)
    # ge = ge.rename(columns={"CELL": "CancID"})
    ge = ge.rename(columns=rename_id_col_names)
    ge = ge[ge[canc_col_name].isin(rsp[canc_col_name].unique())].reset_index(drop=True)

    # Impute
    if sum(ge.isna().sum() > 0):
        raise NotImplementedError("Need to be implemented here!")

    ge_fname = "".join(args.cell_fea)
    metadata["GE_FNAME"] = ge_fname
    # ge_fname = "_".join(args.cell_fea + args.src)
    ge.to_parquet(x_outdir/f"{ge_fname}.parquet", index=False)
    ge.to_csv(x_outdir/f"{ge_fname}.csv", index=False)

    # ----------------
    # X data - drug descriptors/features
    # ----------------
    # import ipdb; ipdb.set_trace(context=5)
    def extract_col_subsets(df, fea_id0=5):
        df_smi = df[[drug_col_name, "SMILES"]]
        df_meta = df.iloc[:, :fea_id0]
        df_fea = pd.concat([ df[drug_col_name], df.iloc[:, fea_id0:] ], axis=1)
        return df_smi, df_meta, df_fea

    fea_id0 = 5

    # Mordred
    dd = load_dd(args.drug_path, impute=False, dropna_th=args.dropna_th,
                 print_fn=print_fn, float_type=np.float32, src=args.src)
    ids = dd[drug_col_name].isin(rsp[drug_col_name].unique())
    dd = dd[ids].reset_index(drop=True)
    dd_smi, dd_meta, dd_fea = extract_col_subsets(dd, fea_id0=fea_id0)

    # ECFP2
    data_path = Path(args.drug_path).parent
    ecfp2 = read_df(data_path/"ecfp2_nbits512")
    ids = ecfp2[drug_col_name].isin(rsp[drug_col_name].unique())
    ecfp2 = ecfp2[ids].reset_index(drop=True)
    ecfp2_smi, ecfp2_meta, ecfp2_fea = extract_col_subsets(ecfp2, fea_id0=fea_id0)

    # # Dragon
    # import ipdb; ipdb.set_trace(context=5)
    # data_path = Path(args.drug_path).parent
    # dragon = read_df(data_path/"combined_drug_descriptors_dragon7_2k")
    # dragon = dragon.rename(columns={"DRUG": "DrugID"})
    # dragon = dragon.rename(columns={s: "dragon_" + s.split("DD_")[1] for s in dragon.columns[1:]})
    # ids = dragon["DrugID"].isin(rsp["DrugID"].unique())
    # dragon = dragon[ids].reset_index(drop=True)
    # dragon_smi, dragon_meta, dragon_fea = extract_col_subsets(dragon, fea_id0=fea_id0)

    # Save
    # mordred_fname = "_".join(args.drug_fea + args.src)
    mordred_fname = "_".join(args.drug_fea)
    metadata["MORDRED_FNAME"] = mordred_fname
    dd_fea.to_parquet(x_outdir/f"{mordred_fname}.parquet", index=False)
    dd_fea.to_csv(x_outdir/f"{mordred_fname}.csv", index=False)

    # smi_fname = "smiles_" + "_".join(args.src)
    smi_fname = "smiles"
    metadata["SMI_FNAME"] = smi_fname
    dd_smi.to_csv(x_outdir/f"{smi_fname}.csv", index=False)

    # ecfp2_fname = "ecfp2_" + "_".join(args.src)
    ecfp2_fname = "ecfp2"
    metadata["ECFP2_FNAME"] = ecfp2_fname
    ecfp2_fea.to_parquet(x_outdir/f"{ecfp2_fname}.parquet", index=False)
    ecfp2_fea.to_csv(x_outdir/f"{ecfp2_fname}.csv", index=False)

    # Impute
    if sum(dd.iloc[:, fea_id0:].isna().sum() > 0):
        raise NotImplementedError("Need to be implemented here!")


    # -----------------------------------------------
    #     Merge data
    # -----------------------------------------------
    # import ipdb; ipdb.set_trace(context=5)
    print_fn('\n{}'.format('-' * 40))
    print_fn('Start merging response with other dfs.')
    print_fn('-' * 40)
    data = rsp

    # Merge with ge
    print_fn('\nMerge with expression (ge).')
    data = pd.merge(data, ge, on=canc_col_name, how='inner')
    groupby_src_and_print(data, col1=canc_col_name, col2=drug_col_name, print_fn=print_fn)
    del ge

    # Merge with dd
    print_fn('\nMerge with descriptors (dd).')
    data = pd.merge(data, dd, on=drug_col_name, how='inner')
    groupby_src_and_print(data, col1=canc_col_name, col2=drug_col_name, print_fn=print_fn)
    del dd

    # Save rsp df for samples that have the features
    # import ipdb; ipdb.set_trace()
    rsp_small = data[rsp.columns]
    # groupby_src_and_print(rsp_small, col1="CancID", col2="DrugID", print_fn=print_fn)
    rsp_small.to_parquet(y_outdir/f"rsp_with_fea.parquet", index=False)
    rsp_small.to_csv(y_outdir/f"rsp_with_fea.csv", index=False)

    # Sample
    if (args.n_samples is not None):
        print_fn('\nSample the final dataset.')
        if args.flatten:
            data = flatten_dist(df=data, n=args.n_samples, score_name=args.trg_name)
        else:
            if args.n_samples <= data.shape[0]:
                data = data.sample(n=args.n_samples, replace=False, random_state=0)
        print_fn(f'data.shape {data.shape}\n')

    # Memory usage
    print_fn('\nTidy dataframe: {:.1f} GB'.format(sys.getsizeof(data)/1e9))
    for fea_name, fea_prfx in fea_prfx_dct.items():
        cols = [c for c in data.columns if fea_prfx in c]
        aa = data[cols]
        mem = 0 if aa.shape[1] == 0 else sys.getsizeof(aa)/1e9
        print_fn('Memory occupied by {} features: {} ({:.1f} GB)'.format(
            fea_name, len(cols), mem))

    # Plot histograms of target variables
    plot_rsp_dists(data, rsp_cols=rsp_cols, savepath=misc_outdir/'rsp_dists.png')

    # -----------------------------------------------
    #   Save data
    # -----------------------------------------------
    # Save data
    print_fn('\nSave dataframe.')
    print_fn("data shape: {}".format(data.shape))
    # fname = create_basename(args)
    fname = "df"
    fpath = misc_outdir/(fname + '.parquet')
    data.to_parquet(fpath)

    # Load data
    print_fn('Load dataframe.')
    print_fn("data shape: {}".format(data.shape))
    data_fromfile = pd.read_parquet(fpath)

    # Check that the saved data is the same as original one
    print_fn(f'Loaded df is same as original: {data.equals(data_fromfile)}')

    print_fn('\n{}'.format('-' * 70))
    print_fn(f'Dataframe filepath:\n{fpath.resolve()}')
    print_fn('-' * 70)


    # -----------------------------------------------
    #   Data splits
    # -----------------------------------------------
    # import ipdb; ipdb.set_trace(context=5)
    print_fn('\nGenerate data splits.')

    get_val_set = True

    np.random.seed(0)
    idx_vec = np.random.permutation(data.shape[0])

    from sklearn.model_selection import KFold
    splitter = KFold(n_splits=10, shuffle=False, random_state=None)
    # Split data (D) into tr (T0) and te (E)
    tr_dct = {}
    te_dct = {}
    # for tr, te in splitter.split(X=idx_vec, y=y_vec, groups=None):
    # idx_vec = range(data.shape[0])
    for i, (tr_id, te_id) in enumerate(splitter.split(X=idx_vec, y=None, groups=None)):
        print(i)
        tr_id = idx_vec[tr_id]
        te_id = idx_vec[te_id]
        tr_dct[i] = tr_id
        te_dct[i] = te_id
        # tr_df = data.loc[idx_vec[tr_id]]
        # te_df = data.loc[idx_vec[te_id]]

        # Generate val set ids from train set ids
        if get_val_set:
            tr_size = len(tr_id) - len(te_id)
            vl_id = tr_id[tr_size:]
            tr_id = tr_id[:tr_size]
            assert len(tr_id) + len(vl_id) + len(te_id) == len(idx_vec), "Size do not match after splitting."
        
            # Make sure that indices do not overlap
            assert len( set(tr_id).intersection(set(vl_id)) ) == 0, 'Overlapping indices btw tr and vl'
            assert len( set(tr_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and te'
            assert len( set(vl_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw vl and te'
            
            # Print split ratios
            print_fn('Train samples {} ({:.2f}%)'.format( len(tr_id), 100*len(tr_id)/data.shape[0] ))
            print_fn('Val   samples {} ({:.2f}%)'.format( len(vl_id), 100*len(vl_id)/data.shape[0] ))
            print_fn('Test  samples {} ({:.2f}%)'.format( len(te_id), 100*len(te_id)/data.shape[0] ))

        # digits = len(str(n_splits))
        seed_str = str(i) # f"{seed}".zfill(digits)
        output = 'split_' + seed_str 
        
        np.savetxt( split_outdir/f'{output}_tr_id', tr_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n' )
        np.savetxt( split_outdir/f'{output}_te_id', te_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n' )

        if get_val_set:
            np.savetxt( split_outdir/f'{output}_vl_id', vl_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n' )

        # Load ids and check consistency
        with open(split_outdir/f'{output}_tr_id') as f:
            t1 = [int(line.rstrip()) for line in f]
        with open(split_outdir/f'{output}_te_id') as f:
            t2 = [int(line.rstrip()) for line in f]
        assert sum(t1 == tr_id) == len(tr_id), "Number of ids missmatch."
        assert sum(t2 == te_id) == len(te_id), "Number of ids missmatch."

        if get_val_set:
            with open(split_outdir/f'{output}_vl_id') as f:
                t3 = [int(line.rstrip()) for line in f]
            assert sum(t3 == vl_id) == len(vl_id), "Number of ids missmatch."

        np.savetxt( split_sorted_outdir/f"{output}_tr_id", sorted(tr_id), fmt="%d", delimiter="", newline="\n" )
        np.savetxt( split_sorted_outdir/f"{output}_te_id", sorted(te_id), fmt="%d", delimiter="", newline="\n" )

        if get_val_set:
            np.savetxt( split_sorted_outdir/f"{output}_vl_id", sorted(vl_id), fmt="%d", delimiter="", newline="\n" )

        ## LC data splits
        np.savetxt( lc_split_outdir/f"{output}_tr_id", sorted(tr_id), fmt="%d", delimiter="", newline="\n" )
        np.savetxt( lc_split_outdir/f"{output}_vl_id", sorted(vl_id), fmt="%d", delimiter="", newline="\n" )
        np.savetxt( lc_split_outdir/f"{output}_te_id", sorted(te_id), fmt="%d", delimiter="", newline="\n" )

        # lc_step_scale = "log"
        # lc_sizes = 7
        lc_step_scale = args.lc_step_scale
        lc_sizes = args.lc_sizes
        assert args.lc_min_size < data.shape[0], "Learning curve lc_min_size must be smaller than the available training set size."
        lc_min_size = args.lc_min_size
        if args.lc_max_size is None:
            lc_max_size = len(tr_id)
        # from learningcurve.lrn_crv import LearningCurve
        # lc_obj = LearningCurve(X=None, Y=None, meta=None, **lc_init_args)
        # pw = np.linspace(0, self.lc_sizes-1, num=self.lc_sizes) / (self.lc_sizes-1)
        # m = self.lc_min_size * (self.lc_max_size/self.lc_min_size) ** pw
        # m = np.array( [int(i) for i in m] ) # cast to int
        pw = np.linspace(0, lc_sizes-1, num=lc_sizes) / (lc_sizes-1)
        m = lc_min_size * (lc_max_size/lc_min_size) ** pw
        m = np.array([int(i) for i in m])  # cast to int
        tr_sizes = m

        # LC subsets
        ids = tr_id
        # ids = sorted(tr_id)  # TODO: should this be sorted?
        for i, sz in enumerate(tr_sizes):
            aa = ids[:sz]
            # aa.to_csv(outdir/f"train_sz_{i+1}.csv", index=False)
            np.savetxt( lc_split_outdir/f"{output}_tr_sz_{i}_id", sorted(aa), fmt="%d", delimiter="", newline="\n" )

    print_fn('\nDone with splits.')


    # -----------------------------------------------
    #   LC data splits
    # -----------------------------------------------
    # Note! We will create the ids for learning curve from the generated data splits
    # import ipdb; ipdb.set_trace(context=5)
    # print_fn('\nGenerate data splits for learning curve.')

    # split_files = list(split_outdir.glob("split_*_id"))
    # # Iter over split_*_tr_id files
    # for fname in [f for f in split_files if "tr_" in f.name]:
    #     # Check that vl and te splits are also available
    #     split_id = fname.split("_")[1]
    #     if (f"split_{split_id}_vl_id" not in split_files) and f"split_{split_id}_tr_id" not in split_files:
    #         continue

    #     # Copy vl and te files to LC folder
    #     import shutil
    #     dest = shutil.copyfile(split_outdir/f"split_{split_id}_vl_id", lc_split_outdir/f"split_{split_id}_vl_id")
    #     dest = shutil.copyfile(split_outdir/f"split_{split_id}_te_id", lc_split_outdir/f"split_{split_id}_te_id")

    #     # Create tr split sizes
    #     # TODO


    # -----------------------------------------------
    #   Train model
    # -----------------------------------------------
    # import ipdb; ipdb.set_trace(context=5)
    train = False
    # train = True

    if train:
        print_fn('\nTrain model.')

        # Split data
        data = data.sample(frac=1.0).reset_index(drop=True)
        te_size = 0.2
        # tr_sz = int(data.shape[0] * 0.8)
        # vl_sz = int(data.shape[0] * 0.9)
        tr_sz = int(data.shape[0] * (1 - 2 * te_size))
        vl_sz = int(data.shape[0] * (1 - te_size))
        tr_data = data[:tr_sz]
        vl_data = data[tr_sz:vl_sz]
        te_data = data[vl_sz:]

        # Get features (x), target (y), and meta
        from ml.data import extract_subset_fea
        fea_list = ["ge", "mordred"]
        fea_sep = "_"
        trg_name = "AUC"
        # xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep=fea_sep)
        # meta = data.drop(columns=xdata.columns)
        # ydata = meta[[trg_name]]
        # del data
        xtr, ytr = extract_subset_fea(tr_data, fea_list=fea_list, fea_sep=fea_sep), tr_data[[trg_name]]
        xvl, yvl = extract_subset_fea(vl_data, fea_list=fea_list, fea_sep=fea_sep), vl_data[[trg_name]]
        xte, yte = extract_subset_fea(te_data, fea_list=fea_list, fea_sep=fea_sep), te_data[[trg_name]]
        assert xtr.shape[0] == ytr.shape[0], "Size missmatch."
        assert xvl.shape[0] == yvl.shape[0], "Size missmatch."
        assert xte.shape[0] == yte.shape[0], "Size missmatch."

        # Scale
        # from ml.scale import scale_fea
        # xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])

        # Train model
        # import ipdb; ipdb.set_trace(context=5)
        import lightgbm as lgb
        ml_init_args = {'n_estimators': 100, 'max_depth': -1, 'learning_rate': 0.1,
                        'num_leaves': 31, 'n_jobs': 8, 'random_state': None}
        # ml_fit_args = {'verbose': False, 'early_stopping_rounds': 10}
        ml_fit_args = {'verbose': True, 'early_stopping_rounds': 10}
        ml_fit_args['eval_set'] = (xvl, yvl)
        model = lgb.LGBMRegressor(objective='regression', **ml_init_args)
        model.fit(xtr, ytr, **ml_fit_args)

        # Predict
        yte_prd = model.predict(xte)
        y_pred = yte_prd
        y_true = yte.values.squeeze()

        # Scores
        import sklearn
        from scipy.stats import pearsonr, spearmanr
        scores = {}
        scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
        scores['mean_absolute_error']   = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['spearmanr'] = spearmanr(y_true, y_pred)[0]
        scores['pearsonr'] = pearsonr(y_true, y_pred)[0]
        print_fn(scores)

        print_fn('\nDone with training.')


    # Save metadata
    # json_object = json.dumps(vars(args), indent=4)
    # with open(outdir/"args.json", "w") as outfile:
    #     outfile.write(json_object)

    # -------------------------------------------------------
    print_fn('\nRuntime: {:.1f} mins'.format((time()-t0)/60))
    print_fn('Done.')
    lg.stop_logger()


def main(args):
    args = parse_args(args)
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
