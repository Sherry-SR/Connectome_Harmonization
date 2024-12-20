#! /usr/bin/env python
import os 
import sys 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

cohort = pd.read_csv('Lists/cohort.csv').set_index('Subject')
sites = ['PNC','TOB']
nedges = 86
N = int(nedges*(nedges-1)/2)
filename_template = 'Connectomes/{1}/{0}_{1}_connmat.txt'
try:
    type = sys.argv[1]
except:
    type = 'desikan'
exists = np.array([os.path.exists(filename_template.format(s,type)) for s in cohort.index])
cohort = cohort.loc[exists,:]
print(cohort)

def load_matrices():
    matfiles = [ filename_template.format(s, type) for s in cohort.index ]
    mats = np.asarray([np.loadtxt(f)[np.triu_indices(nedges,1)] for f in matfiles])
    print(mats.shape)
    return matfiles, mats

def ut_to_square(array, fill=0):
    ret = np.zeros((nedges, nedges))+fill
    ret[np.triu_indices(nedges, 1)] = array
    ret = ret.transpose()
    ret[np.triu_indices(nedges, 1)] = array
    return ret

def nonzero_mask(mats):
    # return boolean array of shape (N,) whether variance is > 0
    var_ = np.var(mats, axis=0)
    return (var_ > 0)
    
matfiles, mats = load_matrices()
intersection_mask = np.prod((mats>0)*1, axis=0) > 0 
#intersection_mask = np.var(mats, axis=0) > 0 
where = np.where(intersection_mask)[0]
W = len(where)
data = mats[:,where]

outdir="Connectomes/{0}_harmonized".format(type)
os.makedirs(outdir, exist_ok=True)
np.savetxt(outdir+'/intersection_mask.txt', intersection_mask, fmt="%d")

# Standardize Age 
cohort['Age'] = (cohort['Age'] - np.mean(cohort['Age']))/np.std(cohort['Age'])
# Initialize harmonized data 
harmonize_cohort = cohort[['Site','Age','Sex']]
harmonize_cohort.loc[:,'Site'] = cohort['Site'].unique()[0]
harmonized = np.zeros(data.shape)
harmonized_connmats = np.zeros(mats.shape)
# # # # # # # # # # # # # # # # # # # # # # # # #
# Canonical link of Gamma is inverse 
# https://www.sagepub.com/sites/default/files/upm-binaries/21121_Chapter_15.pdf
# The default link for the Gamma family is the inverse link. 
# We intentionally added the Warning. Even though the inverse 
# power link is the canonical link function for gamma, it doesn't 
# force the mean to be in the support of the distribution > 0. 
# As a consequence there might be numerical problems for some datasets.
# A common solution is to use the log link instead of inverse power link, 
# which will force the mean prediction to be strictly positive.
# https://github.com/statsmodels/statsmodels/issues/3316

link = sm.genmod.families.links.log
ref_site="TOB"
other_sites=cohort.loc[cohort['Site'] != ref_site,'Site'].unique()
formula = 'Edge ~ C(Site, Treatment(reference="{}")) + Age + np.power(Age,2) + Sex'.format(ref_site)
epsilon = 1e-6
results = pd.DataFrame()

_prev = 0
for i in range(W):
    if int(100*i/(W)) > _prev:
        print(i+1, '/', W, '(', _prev, " % )", end='\r')
        _prev = int(100*i/(W))
    try:
        # [Version B] add epsilon (1e-6) to edge values before gamma estimation and substract it after harmonization
        y = data[:,i] + epsilon
        cohort['Edge'] = y
        model = smf.glm(formula=formula, data=cohort, family=sm.families.Gamma(link=link))
        res = model.fit() 
            
        for n,k in enumerate(res.params.keys()):
            low, high = res.conf_int().loc[k, :]
            results.loc[i, 'coeff_{}'.format(k)] = res.params[k]
            results.loc[i, 'CI-low_{}'.format(k)] = low
            results.loc[i, 'CI-high_{}'.format(k)] = high
            results.loc[i, 't_{}'.format(k)] = res.tvalues[k]
            results.loc[i, 'p_{}'.format(k)] = res.pvalues[k]
            results.loc[i, 'q_{}'.format(k)] = 1 # FDR applied after 
        results.loc[i,'AIC'] = res.aic
        results.loc[i,'BIC'] = res.bic
        results.loc[i,'Pearson-chi2'] = res.pearson_chi2
        results.loc[i,'nb_zeros'] = (y == 0).sum()
        # harmonize out site 
        
        # To harmonize to Site 0 
        # Divide data in Site 1 by exp(Site coeff from GLM)
        # equivalent to subtracting w/ log transformation 
        for site in other_sites:
            site_colname = 'C(Site, Treatment(reference="{0}"))[T.{1}]'.format(ref_site, site)
            site_coeff = res.conf_int().loc[site_colname,:].mean()
            edge_site = y[cohort['Site'] == site]
            harmonized_edge = edge_site / np.exp(site_coeff)
            # put into harmonized data array 
            harmonized[cohort['Site'] == site, i] = harmonized_edge - epsilon
        # put orig data for site0 into data array 
        harmonized[cohort['Site'] == ref_site, i] = y[cohort['Site'] == ref_site] - epsilon
      
    except Exception as e:
        print(str(e))
        
results.to_csv(outdir+'/harmonization_glm_results_{0}.csv'.format(type))

for i, subject in enumerate(cohort.index):
    harmonized_connmat_ut = np.zeros((N,))
    harmonized_connmat_ut[where] = harmonized[i,:]
    harmonized_connmat = ut_to_square(harmonized_connmat_ut)
    np.savetxt(outdir+'/{0}_{1}_harmonized_connmat.txt'.format(subject, type), harmonized_connmat)
