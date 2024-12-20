#! /usr/bin/env python

import numpy as np
from statsmodels.genmod.families import Gamma, links
import statsmodels.formula.api as smf

def load_connectomes(cohort, filename_template):
    matfiles = [ filename_template.format(s=s) for s in cohort.index ]
    mats = np.asarray([np.loadtxt(f) for f in matfiles])
    return matfiles, mats

def ut_to_square(array, fill=0):
    N = len(array)
    nedges = 0.5*(np.sqrt(8*N + 1)+1)
    if not nedges.is_integer():
        raise ValueError('Input array cannot be converted to square matrix')
    nedges = int(nedges)
    ret = np.zeros((nedges, nedges)) + fill
    ret[np.triu_indices(nedges, 1)] = array
    ret = ret.transpose()
    ret[np.triu_indices(nedges, 1)] = array
    return ret
    
def square_to_ut(matrix):
    nedges = matrix.shape[0]
    return matrix[np.triu_indices(nedges,1)]

def nonzero_variance_mask(mats):
    # return boolean array of shape (N,) whether variance is > 0
    var_ = np.var(mats, axis=0)
    return (var_ > 0)
    
def match_harmonize_connectomes(cohort, matrices, covars_formula='', sitecol='Site', ref_site=None):
    if ref_site is None:
        all_sites = np.unique(cohort[sitecol])
        ref_site = str(all_sites[0])
    other_sites=cohort.loc[cohort[sitecol] != ref_site,sitecol].unique()
    
    formula = 'Edge ~ C(Site, Treatment(reference="{0}")) + {1}'.format(ref_site,covars_formula)
    print('formula:', formula)
    # upper triangular 
    data = np.array([square_to_ut(mat) for mat in matrices])
    
    # change to within-site variance mask 
    # variance must be zero within site for every site. 
    nzmask = nonzero_variance_mask(data)  # shape (nedges,)
    W = nzmask.sum()
    harmonized = np.zeros(data.shape)
    site_mult = np.ones(data.shape)
    # results = pd.DataFrame()
    _prev = 0
    epsilon = 1e-6

    # for i in range(W):
    for i in np.where(nzmask)[0]:
        if int(100*i/(W)) > _prev:
            print(i+1, '/', W, '(', _prev, " % )", end='\r')
            _prev = int(100*i/(W))
        try:
            y = data[:,i] + epsilon
            positive_mask = y>0
            
            cohort['Edge'] = y
            model = smf.glm(formula=formula, data=cohort.loc[positive_mask,:], family=Gamma(link=links.log))
            res = model.fit(method="lbfgs") 
            
            # To harmonize to Site 0 
            # Divide data in Site 1 by exp(Site coeff from GLM)
            # equivalent to subtracting w/ log transformation 
            for site in other_sites:
                site_colname = 'C(Site, Treatment(reference="{0}"))[T.{1}]'.format(ref_site, site)
                site_coeff = res.conf_int().loc[site_colname,:].mean()
                edge_site = y[cohort['Site'] == site]
                # edges with zero values will remain zero after harmonization, other edges will be adjusted for site effects
                harmonized_edge = edge_site / np.exp(site_coeff)
                # put into harmonized data array, substract epsilon for Method 'epsilon' (epsilon=0 otherwise)
                harmonized[cohort['Site'] == site, i] = harmonized_edge - epsilon
                site_mult[cohort['Site'] == site, i] = 1.0/np.exp(site_coeff)
            # put orig data for site0 into data array, substract epsilon for Method 'epsilon' (epsilon=0 otherwise)
            harmonized[cohort['Site'] == ref_site, i] = y[cohort['Site'] == ref_site] - epsilon
        except Exception as e:
            print('exception encountered in edge', i)
            print(str(e))
    
    # convert UT to square 
    harmonized_matrices = np.asarray([ ut_to_square(x) for x in harmonized])
    site_mult_matrices = np.asarray([ ut_to_square(r) for r in site_mult])
    return harmonized_matrices, site_mult_matrices

