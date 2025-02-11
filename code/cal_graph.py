# import required packages
import pickle
import os
import numpy as np
import pandas as pd
import bct
import statsmodels.formula.api as smf
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

def calculate_degree_intraHemisphere(connectome_,hemisphereMaps,sign="positive"):
    connectome=np.copy(connectome_)
    np.fill_diagonal(connectome,0)
    numNodes = len(connectome)
    degree = np.zeros(numNodes)
    if(sign=="positive"):
        for i in range(numNodes):
            degree[i] = np.sum(connectome[i][hemisphereMaps==hemisphereMaps[i]]>0)
    elif(sign=="negative"):
        for i in range(numNodes):
            degree[i] = np.sum(connectome[i][hemisphereMaps==hemisphereMaps[i]]<0)
    return degree

def calculate_degree_interHemisphere(connectome_,hemisphereMaps,sign="positive"):
    connectome=np.copy(connectome_)
    np.fill_diagonal(connectome,0)
    numNodes = len(connectome)
    degree = np.zeros(numNodes)
    if(sign=="positive"):
        for i in range(numNodes):
            degree[i] = np.sum(connectome[i][hemisphereMaps!=hemisphereMaps[i]]>0)
    elif(sign=="negative"):
        for i in range(numNodes):
            degree[i] = np.sum(connectome[i][hemisphereMaps!=hemisphereMaps[i]]<0)
    return degree

def calculate_strength_intraHemisphere(connectome_,hemisphereMaps,sign="positive"):
    connectome=np.copy(connectome_)
    np.fill_diagonal(connectome,0)
    numNodes = len(connectome)
    strength = np.zeros(numNodes)
    if(sign=="positive"):
        for i in range(numNodes):
            mask=connectome[i][hemisphereMaps==hemisphereMaps[i]]>0
            strength[i] = np.sum(connectome[i][hemisphereMaps==hemisphereMaps[i]][mask])
    elif(sign=="negative"):
        for i in range(numNodes):
            mask=connectome[i][hemisphereMaps==hemisphereMaps[i]]<0
            strength[i] = np.sum(connectome[i][hemisphereMaps==hemisphereMaps[i]][mask])
    return strength
    
def calculate_strength_interHemisphere(connectome_,hemisphereMaps,sign="positive"):
    connectome=np.copy(connectome_)
    np.fill_diagonal(connectome,0)
    numNodes = len(connectome)
    strength = np.zeros(numNodes)
    if(sign=="positive"):
        for i in range(numNodes):
            mask=connectome[i][hemisphereMaps!=hemisphereMaps[i]]>0
            strength[i] = np.sum(connectome[i][hemisphereMaps!=hemisphereMaps[i]][mask])
    elif(sign=="negative"):
        for i in range(numNodes):
            mask=connectome[i][hemisphereMaps!=hemisphereMaps[i]]<0
            strength[i] = np.sum(connectome[i][hemisphereMaps!=hemisphereMaps[i]][mask])
    return strength 

def main():
    parser = argparse.ArgumentParser(description='Calculate graph measures')
    parser.add_argument('-d', '--data_dir', type=str, default='../Data/MATCH', help='path to data')
    parser.add_argument('-o', '--output_dir', type=str, default='../Results/MATCH/', help='path to outputs')
    parser.add_argument('-a', '--atlas', type=str, default='desikan', help='atlas')
    parser.add_argument('-m', '--method', type=str, default='normalized_unharmonized', help='harmonization method')
    parser.add_argument('-c', '--cohort', type=str, default='QCpass', help='cohort name')
    parser.add_argument('-t', '--thresh', type=str, default='None', help='consistency thresholding level')
    args = parser.parse_args()

    cohort = pd.read_csv(os.path.join(args.data_dir, 'cohort', 'cohort_'+args.cohort+'.csv'))
    ROI = pd.read_csv(os.path.join(args.data_dir, 'atlas', args.atlas+'.csv'))
    data = []
    if args.method == 'normalized_unharmonized':
        folder = args.method
        suffix = '_normalized_'+args.atlas+'_connmat.txt'
    elif args.method in ['MATCH_epsilon', 'MATCH_positivemask']:
        folder = 'harmonized_'+args.method+'/'+args.cohort
        suffix = '_normalized_'+args.atlas+'_connmat_MATCH.txt'
    else:
        folder = 'harmonized_'+args.method+'/'+args.cohort
        suffix = '_'+args.atlas+'_'+args.method+'_connmat.txt'
    for subj in cohort['Subject']:
            mat = np.loadtxt(os.path.join(args.data_dir, folder, args.atlas, subj+suffix))
            mat[np.isnan(mat)] = 0
            #if args.method in ['logcombat', 'logcovbat', 'MATCH_epsilon']:
            #    mat = mat + 1e-6
            mask = np.loadtxt(os.path.join(args.data_dir, 'normalized_unharmonized', args.atlas, subj+'_normalized_'+args.atlas+'_connmat.txt')) != 0
            mat = mat * mask
            data.append(mat)

    data = np.array(data)
    dim = data.shape[1]

    output_suffix = '_'+args.cohort+'_'+args.method
    if args.thresh != 'None':
        c = np.abs(np.std(data[cohort['Group']=='TDC'], axis=0, ddof=1) / np.mean(data[cohort['Group']=='TDC'], axis=0))
        c[np.isnan(c)] = c[~np.isnan(c)].max()+1
        consistency_mask = c <= np.percentile(c, 100*float(args.thresh))
        data[:, ~consistency_mask] = 0
        output_suffix = output_suffix + '_thr' + args.thresh
    else:
        consistency_mask = None

    if not os.path.exists(os.path.join(args.output_dir, 'logger')):
        os.makedirs(os.path.join(args.output_dir, 'logger'))
    
    logger = logging.getLogger('Calculate graph measures')
    hdlr = logging.FileHandler(os.path.join(args.output_dir, 'logger', output_suffix+'.log'), 'w+')
    formatter = logging.Formatter('%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    logger.info(' '.join([args.data_dir, args.output_dir, args.atlas, args.method, args.cohort, args.thresh]))

    # edgewise analysis
    groups = np.unique(cohort['Group'])
    for group in groups:
        logger.info('edgewise analysis for '+group+'...')
        ftest_f = np.zeros((dim, dim))
        ftest_p = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i+1, dim):
                cohort['Edge'] = data[:, i, j]
                mdl = smf.ols('Edge ~ Site + Age + np.power(Age,2) + Sex', data=cohort[cohort['Group']==group])
                mdl2 = smf.ols('Edge ~ Age + np.power(Age,2) + Sex', data=cohort[cohort['Group']==group])
                res = mdl.fit()
                res2 = mdl2.fit()
                ftest_f[i, j], ftest_p[i, j], _ = res.compare_f_test(res2)
        ftest_f = ftest_f + ftest_f.T
        ftest_p = ftest_p + ftest_p.T
        with open(os.path.join(args.output_dir, 'edge_' + args.atlas + output_suffix + '_'+ group +'.pickle'), 'wb') as handle:
            pickle.dump([ftest_f, ftest_p, consistency_mask], handle)

    # graph toplogical measures
    nodal_stats = {'Subject':[], 'node_degree':[], 'node_strength':[],
                'node_betweenness_centrality':[], 'eccentricity':[],
                'local_efficiency':[], 'clustering_coefficient':[]}

    global_stats = {'Subject':[], 'degree':[], 'degree_intraHemisphere':[], 'degree_interHemisphere':[],
                    'strength':[], 'strength_intraHemisphere':[], 'strength_interHemisphere':[],
                'characteristic_path_length':[], 'assortativity':[],
                'global_efficiency':[], 'modularity_global':[]}

    for i in range(len(data)):
        if (i % 10) == 0:
            logger.info('processing '+str(i)+'/'+str(len(data)))
        nodal_stats['Subject'].append(cohort['Subject'].iloc[i])
        global_stats['Subject'].append(cohort['Subject'].iloc[i])

        connectome = data[i].copy()
        np.fill_diagonal(connectome, 0)
        connectome = (connectome + connectome.transpose())/2.0
        connectome_norm = bct.utils.weight_conversion(connectome,"normalize")
        connectome_length_norm = bct.utils.weight_conversion(connectome_norm,"lengths")
        connectome_dist_length_norm = bct.distance_wei(connectome_length_norm)[0]
        hemisphereMaps = ROI['Hemisphere'] == 'R'

        bct_charpath_results = bct.charpath(connectome_dist_length_norm)
        # nodal topological measures
        nodal_stats['node_degree'].append(bct.degrees_und(connectome))
        nodal_stats['node_strength'].append(bct.strengths_und(connectome))
        nodal_stats['node_betweenness_centrality'].append(bct.betweenness_wei(connectome_length_norm)/float((dim-1)*(dim-2)))
        nodal_stats['eccentricity'].append(bct_charpath_results[2])
        nodal_stats['local_efficiency'].append(bct.efficiency_wei(connectome_norm,local=True))
        nodal_stats['clustering_coefficient'].append(bct.clustering_coef_wu(connectome_norm))

        # global topological measures
        global_stats['degree'].append(bct.degrees_und(connectome).sum()/2)
        global_stats['degree_intraHemisphere'].append(calculate_degree_intraHemisphere(connectome, hemisphereMaps).sum()/2)
        global_stats['degree_interHemisphere'].append(calculate_degree_interHemisphere(connectome, hemisphereMaps).sum()/2)
        global_stats['strength'].append(bct.strengths_und(connectome).sum()/2)
        global_stats['strength_intraHemisphere'].append(calculate_strength_intraHemisphere(connectome, hemisphereMaps).sum()/2)
        global_stats['strength_interHemisphere'].append(calculate_strength_interHemisphere(connectome, hemisphereMaps).sum()/2)
        global_stats['characteristic_path_length'].append(bct_charpath_results[0])
        global_stats['assortativity'].append(bct.assortativity_wei(connectome_norm,flag=0))
        global_stats['global_efficiency'].append(bct_charpath_results[1])
        global_stats['modularity_global'].append(bct.modularity_louvain_und(connectome)[1])

    nodal_stats = pd.DataFrame(nodal_stats)
    nodal_stats.to_csv(os.path.join(args.output_dir, 'nodal_' + args.atlas + output_suffix + '.csv'))
    global_stats = pd.DataFrame(global_stats)
    global_stats.to_csv(os.path.join(args.output_dir, 'global_' + args.atlas + output_suffix + '.csv'))
    with open(os.path.join(args.output_dir, 'measures_' + args.atlas + output_suffix + '.pickle'), 'wb') as handle:
        pickle.dump([nodal_stats, global_stats], handle)

if __name__ == "__main__":
    main()
