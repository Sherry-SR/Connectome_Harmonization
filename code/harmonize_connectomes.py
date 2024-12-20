#! /usr/bin/env python

import os 
import sys
import argparse 
import re 
import pandas as pd
import numpy as np
from match import load_connectomes, match_harmonize_connectomes

DESCRIPTION = '''
    Run MATCH harmonization algorithm on connectomes. 
'''
PROTOCOL_NAME='MATCH' 

def parseCommandLine(argv=None):
    # create arguments for the inputs
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('-c', type=str, dest='cohort', required=True,
                help='Path to a cohort csv with subjectIDs in the first column')
    p.add_argument('-f', type=str, dest='filename_template', required=True,
                help='Template for connectome paths with {s} in place of subject IDs')
    p.add_argument('-o',  type=str, dest='outdir', required=True, 
                help='The output directory.')
    p.add_argument('-V', '-covariates', type=str, default='', dest='formula', 
                help='Patsy-like formula for the covariates (e.g "Age+Sex").')
    p.add_argument('-S', '-site', type=str, default='Site', dest='site', 
                help='Column name encoding the site or study variable. Default is "Site".')
    p.add_argument('-R', '-refsite', type=str, default=None, dest='refsite', 
                help='Site label to be used as the reference site.')
    p.add_argument('--debug', action='store_true', dest='debug',
                    required=False, default=False, 
                    help='Debug mode'
                    )
    p.add_argument('--logfile', action='store', metavar='log', dest='logfile', 
                    type=str, required=False, default=None, 
                    help='A log file. If not provided will print to stderr.'
                    )
    args = p.parse_args(argv)
    return args

def prepare_cohort(cohortcsv, filename_template, site, formula, na_values=['.','NAN','NaN','nan','Nan','NA','na','n/a','N/A',' ']):
    """
    Prepare cohort for harmonization/analysis. 
    Loads a cohort csv file with pandas, and 
    reduces to entries with existing data paths 
    and requisite site and covariate information. 
    
    Parameters
    ----------
    cohortcsv : str
        Path to a comma-separated spreadsheet text file 
    filename_template : str
        Path to data with {s} in place of subject ID
    site : str
        The column name corresponding to the site, 
        scanner or batch information 
    formula : str 
        A patsy-like formula for covariates to control 
        for during harmonization 
    na_values : list
        A list of values encoding NaN in the cohort 

    Returns
    -------
    pandas.DataFrame
        The filtered and reduced cohort dataframe

    Raises
    ------
    ValueError
        When site column is included in formula 
    """

    def split_formula(formula):
        # return a list of the words in the formula 
        words = list(filter(lambda c: bool(c), re.split('\)|\(|\+|\*|\:|\ ', formula)))
        return words 
    # Read cohort, grab only columns needed, remove any missing data
    cohort = pd.read_csv(cohortcsv, na_values=na_values, index_col=0)
    # filter to data exist 
    exists = np.asarray([os.path.exists(filename_template.format(s=s)) for s in cohort.index])
    # Get column names from formula 
    covariates = split_formula(formula)
    if site in covariates:
        raise ValueError('Site column ({0}) cannot be in covariate formula'.format(site))
    cohort = cohort.loc[exists, [site]+covariates]
    cohort = cohort.dropna()
    return cohort 

def main(argv):
    args = parseCommandLine(argv) 
    cohort = prepare_cohort(args.cohort, args.filename_template, args.site, args.formula)
    subjects = list(cohort.index)
    filenames, connectomes = load_connectomes(cohort, args.filename_template)
    # make outdir 
    os.makedirs(args.outdir, exist_ok=True)
    harmonized_connmats, site_mult_connmats = match_harmonize_connectomes(cohort, connectomes, covars_formula=args.formula, 
        sitecol=args.site, ref_site=args.refsite)
        
    for s, c, d, f in zip(subjects, harmonized_connmats, site_mult_connmats, filenames):
        outf = os.path.join(args.outdir, os.path.basename(f[:-4]+'_MATCH.txt'))
        print(outf)
        np.savetxt(outf, c, fmt='%0.8e')
        outf = os.path.join(args.outdir, os.path.basename(f[:-4]+'_SITE-MULTIPLIER.txt'))
        np.savetxt(outf, d, fmt='%0.8e')
 
if __name__ == '__main__':
    main(sys.argv[1:])
