import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

import sklearn
import numpy as np
import pandas as pd
from glob import glob

from sklearn import metrics
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt

# fpath = Path(__file__).resolve().parent
# sys.path.append( str(fpath/'../src') ) # must convert to str
from learningcurve import lc_plots

# from fit import fit_params, biased_powerlaw
from post_process.fit import fit_params, biased_powerlaw

from post_process.nls_lm import fit_model  #, fit_params, biased_powerlaw

# Plot params
legend_fontsize = 12


def drop_bad_r2(df):
    """ Remove runs with negative R2. """
    if df is None:
        return None
    else:
        print(df.shape)
        df['id'] = [str(sz)+'_'+str(r) for sz, r in zip(df['tr_size'], df['run'])]
        aa = df[ df['metric']=='r2' ]
        aa['valid'] = aa['score'] > 0
        ids_valid = aa[ aa['valid']==True ].id.unique()
        aa = df[ df['id'].isin(ids_valid) ]
        print(aa.shape)
    return aa


def print_count(aa):
    # aa = nn0_rnd
    aa['one'] = 1
    aa = aa[(aa['set']=='te') & (aa['metric']=='mean_absolute_error')]
    display(aa.groupby(['tr_size']).agg({'one': 'sum'}).sort_values('tr_size').reset_index())

    
def load_data(path, tr_set='te'):
    """ Load scores. """
    if path is None:
        return None
    else:
        df = pd.read_csv(path);
        df.rename(columns={'split': 'run'}, inplace=True)
        if tr_set != 'all':
            df = df[ df['set'] == tr_set ].reset_index(drop=True)
        return df


def load_data_hpc(path, tr_set='te'):
    """ Load scores from the runs on Summit HPC. """
    df = pd.read_csv(path);
    df.rename(columns={'split': 'run'}, inplace=True)
    if tr_set != 'all':
        df = df[ df['set'] == tr_set ].reset_index(drop=True)
    return df


def fit_data(df, x_fit_mn=0, x_fit_mx=None, met='mean_absolute_error',
             method='linear', only_median=True):
    """ Return df with the median computed for the met. """
    if df is None:
        return None
    
    df = df[ df['metric']==met ].reset_index(drop=True)
    
    if only_median:
        df['y'] = df['score']
        df = df[df['metric']==met].groupby('tr_size').agg({'y': 'median'}).reset_index()
    
    dfit = subset_data(df, col='tr_size', x_mn=x_fit_mn, x_mx=x_fit_mx)
    # dfit = add_weight_col( dfit, binomial=False )
    dfit = add_weight_col( dfit, method=method )
    return dfit


def add_weight_col(df, method='linear'):
    """ ... """
    if method == 'const':
        df['w'] = 1
        
    elif method == 'linear':
        df['w'] = df['tr_size'] / df['tr_size'].max()
        
    elif method == 'binomial':
        df['w'] = df['tr_size'] / ( df['y'] * (1-df['y']) )
        
    return df


def subset_data(df, col='tr_size', x_mn=None, x_mx=None):
    """ Subset df based on range. """
    if x_mn is not None:
        df = df[ df[col] >= x_mn ].reset_index(drop=True)
    if x_mx is not None:
        df = df[ df[col] <= x_mx ].reset_index(drop=True)
    return df


def pwr_law(x, a, b, c):
    y = a * x**(b) + c
    return y


def calc_fit(x, coefs):
    """ ... """
    coefs = coefs.reset_index(drop=True)
    args = { coefs.loc[i, 'coef']: coefs.loc[i, 'est'] for i in range(len(coefs)) }
    args.update({'x': x})
    y = pwr_law( **args )
    return y


def calc_gof(y, yfit):
    from sklearn import metrics
    gof = {}
    rmse = sqrt( metrics.mean_squared_error(y, yfit) )
    mae = metrics.mean_absolute_error(y, yfit)
    r2 = metrics.r2_score(y, yfit)
    gof['rmse'] = rmse
    gof['mae'] = mae
    gof['r2'] = r2
    return gof


def plot_lc_model(df, src_name, label, **kwargs):
    """ Plor LC data. """
    if df is None:
        return None
    kwargs.update({'title': f'{src_name}; {label}'})
    ax = lc_plots.plot_lc_single_metric(df, **kwargs);
    ax.legend(frameon=True, fontsize=legend_fontsize, loc='best')
    ax.grid(False)
    return ax


def inv_powerlaw(y, prms):
    vv = ((y - prms['gamma']) / prms['alpha'] ) ** (1/prms['beta'])
    if np.isnan(vv) == False:
        vv = int(vv)
    return vv


def get_score_at_2mK(dfit, prms):
    x_a = 2 * dfit['tr_size'].values[-1]
    return biased_powerlaw(x_a, **prms)
        

def fit_and_ci(dfit, q=0.1):
    """ Fit each run to power-law and return the fitted results.
    Args:
        dfit : 
        q : quantile to compute variability of power law fits
    Returns:
        data_fitted : dfit with a column 'score_fit' added
        ci : df with 4 columns (tr_size, ci_lwr, ci_upr, median)
    """
    dfs = []

    # Iter over runs and fit the LC data
    for run_name in dfit['run'].unique():
        data_fit = dfit[ dfit['run'] == run_name ].sort_values(by='tr_size').reset_index(drop=True)

        # At least 3 fitting points are required
        if len(data_fit) < 4:
            continue

        xf = data_fit['tr_size'].values
        yf = data_fit['score'].values
        w  = data_fit['w'].values

        prms = fit_params(x=xf, y=yf)
        data_fit['score_fit'] = biased_powerlaw(xf, **prms)
        dfs.append(data_fit)
        
        del prms

    # Concat results
    data_fitted = pd.concat(dfs, axis=0).reset_index(drop=True)
    
    # Compute the lower bound of the fit for each tr_size
    ci_lwr = data_fitted[['tr_size', 'score_fit']]
    # ci_lwr = ci_lwr.groupby('tr_size').agg({'score_fit': 'min'}).reset_index().rename(columns={'score_fit': 'ci_lwr'})
    ci_lwr = ci_lwr.groupby('tr_size').agg({'score_fit': lambda x: np.quantile(x, q=q)}).reset_index().rename(columns={'score_fit': 'ci_lwr'})

    # Compute the upper bound the fit for each tr_size
    ci_upr = data_fitted[['tr_size', 'score_fit']]
    # ci_upr = ci_upr.groupby('tr_size').agg({'score_fit': 'max'}).reset_index().rename(columns={'score_fit': 'ci_upr'})
    ci_upr = ci_upr.groupby('tr_size').agg({'score_fit': lambda x: np.quantile(x, q=1-q)}).reset_index().rename(columns={'score_fit': 'ci_upr'})

    # Compute the median the fit for each tr_size
    med = data_fitted[['tr_size', 'score_fit']]
    med = med.groupby('tr_size').agg({'score_fit': 'median'}).reset_index().rename(columns={'score_fit': 'median'})
    
    # Merge
    ci = ci_lwr.merge(ci_upr, on='tr_size')
    ci = ci.merge(med, on='tr_size')
    
    # Fit the median to power-law
    prms_med = fit_params(x=ci['tr_size'], y=ci['median'])
    ci['med_fit'] = biased_powerlaw(ci['tr_size'], **prms_med)
    # prms_med['set'] = 'median'
    
    prms_lwr = fit_params(x=ci['tr_size'], y=ci['ci_lwr'])
    ci['ci_lwr_fit'] = biased_powerlaw(ci['tr_size'], **prms_lwr)
    # prms_lwr['set'] = 'ci_lwr'

    prms_upr = fit_params(x=ci['tr_size'], y=ci['ci_upr'])
    ci['ci_upr_fit'] = biased_powerlaw(ci['tr_size'], **prms_upr)
    # prms_upr['set'] = 'ci_upr'
    
    # prms = pd.DataFrame([prms_med, prms_lwr, prms_upr])
    
    prms = {}
    prms['median'] = prms_med
    prms['ci_lwr'] = prms_lwr
    prms['ci_upr'] = prms_upr

    return data_fitted, ci, prms
    

def plot_lc_fit_ci(data_fitted=None,
                   ci=None,
                   met='mean_absolute_error',
                   xtick_scale='log2', ytick_scale='log2',
                   med_color='k',
                   lc_raw=True,
                   median=True,
                   ci_lines=False,
                   ci_shade=False,
                   errorbars=False,
                   name=None,
                   label=None,
                   title=None,
                   ax=None,
                   xlabels_log=None,
                   **kwargs):
    """ Plot the following:
        1) LC_raw above m_kmin
        2) median across the fits 
        3) power law fit to lower and upper quantiles 

    Args:
        The primary to dfs are data_fitted and ci that come from fit_and_ci.
    """
    lc_pts_args = {'metric_name': met, 'xtick_scale': xtick_scale, 'ytick_scale': ytick_scale, 'alpha': 0.7, 'ls': '', 'marker': '.'}
    med_pts_args = {'metric_name': met, 'xtick_scale': xtick_scale, 'ytick_scale': ytick_scale, 'alpha': 1, 'ls': '', 'marker': 's'}
    med_fit_args = {'metric_name': met, 'xtick_scale': xtick_scale, 'ytick_scale': ytick_scale, 'alpha': 1, 'ls': '-', 'marker': ''}
    ci_pts_args = {'metric_name': met, 'xtick_scale': xtick_scale, 'ytick_scale': ytick_scale, 'alpha': 1, 'ls': '', 'marker': '.'}
    ci_fit_args = {'metric_name': met, 'xtick_scale': xtick_scale, 'ytick_scale': ytick_scale, 'alpha': 0.7, 'ls': '--', 'marker': ''}
    
    if lc_raw and data_fitted is not None:
        df = data_fitted[data_fitted['metric']==met].reset_index(drop=True)
        x = df['tr_size'].values
        y = df['score_fit'].values
        ax = lc_plots.plot_lc(x=x, y=y, ax=ax, **lc_pts_args, color='0.5', label='$y$')

    if ci is not None:
        x = ci['tr_size'].values
    
        if median:
            if name:
                ax = lc_plots.plot_lc(x=ci['tr_size'].values, y=ci['median'].values, ax=ax, **med_pts_args, color=med_color, label=f'{name} data')
                ax = lc_plots.plot_lc(x=ci['tr_size'].values, y=ci['med_fit'].values, ax=ax, **med_fit_args, color=med_color, label=f'{name} fit')
            else:
                ax = lc_plots.plot_lc(x=ci['tr_size'].values, y=ci['median'].values, ax=ax, **med_pts_args, color=med_color, label='$\~y$')
                ax = lc_plots.plot_lc(x=ci['tr_size'].values, y=ci['med_fit'].values, ax=ax, **med_fit_args, color=med_color, label=f'$\~y$ fit')

        if ci_lines:
            ax = lc_plots.plot_lc(x=ci['tr_size'].values, y=ci['ci_lwr'].values, ax=ax, **ci_pts_args, color='b', label='$q_{0.1}$')
            ax = lc_plots.plot_lc(x=ci['tr_size'].values, y=ci['ci_lwr_fit'].values, ax=ax, **ci_fit_args, color='b', label='$q_{0.1}$ fit')
            ax = lc_plots.plot_lc(x=ci['tr_size'].values, y=ci['ci_upr'].values, ax=ax, **ci_pts_args, color='g', label='$q_{0.9}$')
            ax = lc_plots.plot_lc(x=ci['tr_size'].values, y=ci['ci_upr_fit'].values, ax=ax, **ci_fit_args, color='g', label='$q_{0.9}$ fit')

        if ci_shade:
            ax.fill_between(ci['tr_size'].values, y1=ci['ci_lwr_fit'].values, y2=ci['ci_upr_fit'].values, color=med_color, alpha=0.07)

        if errorbars:
            # Doesn't work
            yerr = np.asarray([ci['ci_lwr'].values, ci['ci_upr'].values])
            ax = lc_plots.plot_lc(x=ci['tr_size'].values, y=ci['median'].values, yerr=yerr, ax=ax, **ci_fit_args, color='g', label='xxx')

    if title:
        ax.set_title(title)
    ax.legend(frameon=True, fontsize=12, loc='best')
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=11)
    ax = lc_plots.set_yticks(ax)
    ax = lc_plots.set_xticks(ax, xlabels_log, show_log=True)
    return ax    
    
    
    
# ------------------------------------------------------------------------
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


class FitPwrLaw():
    
    # def __init__(self, xf, yf, w=None):
    def __init__(self, xf, yf, w=None, a: float=1.2, b: float=-0.3, c: float=0.03):
        assert len(xf) == len(yf), 'xf and yf must be equal size.'
        self.xf = xf
        self.yf = yf
        if w is not None:
            assert len(xf) == len(w), 'xf and w must be equal size.'
            self.w = w
        else:
            w = np.ones( (len(self.xf),) )
            
        # New!
        self.a = a
        self.b = b
        self.c = c
            
        self.fit_model()
        # self.fit_params()

        
    def fit_model(self):
    # def fit_model(x: ro.IntVector, y: ro.FloatVector, w: ro.FloatVector):
        """ ... """
        x = ro.IntVector(list(self.xf))
        y = ro.FloatVector(list(self.yf))
        w = ro.FloatVector(list(self.w))

        # script = '\'../fit.R\''
        script = '\'nls_lm.R\''
        ro.r('''source({})'''.format(script))
        fit_nlsLM_power_law = ro.globalenv['fit_nlsLM_power_law']
        # coef_est_r = fit_nlsLM_power_law(x, y, w)  # commented
        coef_est_r = fit_nlsLM_power_law(x, y, w,
                                         a=self.a, b=self.b, c=self.c)  # new!

        # coef_est_py = pandas2ri.ri2py_dataframe(coef_est_r)
        with localconverter(ro.default_converter + pandas2ri.converter):
            coef_est_py = ro.conversion.rpy2py(coef_est_r)

        self.coefs = coef_est_py.reset_index(drop=True)
        self.a = self.coefs.loc[ self.coefs['coef'] == 'a', 'est'].values
        self.b = self.coefs.loc[ self.coefs['coef'] == 'b', 'est'].values
        self.c = self.coefs.loc[ self.coefs['coef'] == 'c', 'est'].values
    
    
    def fit_params(self):
        x = ro.IntVector(list(self.xf))
        y = ro.FloatVector(list(self.yf))
        
        script = '\'fit.R\''
        # script = '\'../fit.R\''
        ro.r('''source({})'''.format(script))
        get_params = ro.globalenv['model_param']
        a, b, c = get_params(x, y)

        # gamma, alpha, beta = get_params(x, y)
        prms = {}
        # prms_dct['alpha'], prms_dct['beta'], prms_dct['gamma'] = b, c-0.5, a

        prms['alpha'] = b
        prms['beta'] = c - 0.5
        prms['gamma'] = a
        
        self.prms = prms
        self.a = self.prms['alpha']
        self.b = self.prms['beta']
        self.c = self.prms['gamma']
        
    
    def calc_fit(self, x=None, x1=None, x2=None):
        """ Calculate the fit. """
        if x is not None:
            y = self.a * x**(self.b) + self.c
            
        elif (x1 is not None) and (x2 is not None):
            x = np.linspace(x1, x2, 50)
            y = self.a * x**(self.b) + self.c
            
        else:
            x = np.linspace(xf.min(), xf.max(), 50)
            y = self.a * x**(self.b) + self.c
            
        return x, y
