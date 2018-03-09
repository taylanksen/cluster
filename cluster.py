#!/usr/bin/env python
"""
------------------------------------------------------------------------
  Given:
    frames.csv 
    k range
    
  creates:
    clusters_k.csv
    clusters_k.png
    scores.csv 

  example:
      $ python -i 'example/test.csv'
------------------------------------------------------------------------
"""
from __future__ import print_function
import csv
import numpy as np
import pandas as pd
import math
from scipy import linalg
from scipy.stats import multivariate_normal
import operator 

#import compare
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn import mixture

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import glob
import os
import sys
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

from collections import defaultdict
from itertools import combinations
import itertools

# you can set the logging level from a command-line option such as:
#   --log=INFO


#-------------------------------------------------------------------------------
class ClusterSearch:
    """ 
    
    public methods:
    :ClusterSearch(inputfile) - specify file containing openface frames
    :feature_search - runs clustering for subsets of specified features, k_range 
    
    :k_search   - runs sklearn kmeans over range of k values
    :gmm_search - runs sklearn gmm over range of k
    :bin_search - counts frequencies for all possible binary subsets
    
    :write_clusters
    :write_scores
    :write_plots
    """
    CONFIDENCE_TOL = 0.90  # only faces with confidence >= will be used
    
    def __init__(s, infile):
        log.info('...loading data')
        if 'pkl' in infile:
            s.df = pd.read_pickle(infile)
        else:
            s.df = pd.read_csv(infile, skipinitialspace=True) 
        
        # remove bad confidence
        if 'confidence' in s.df.columns:
            s.df = s.df[s.df['confidence'] >= ClusterSearch.CONFIDENCE_TOL]
        else:
            log.info.warning('No confidence column, using all datapoints')

        log.info('...data loaded')

            
    #-------------------------------------------
    def feature_search(s, features, k_range, max_dim, f_search):
        """ runs f_search within k_range for all subsets of features up 
        to max_dim. Output results are saved by f_search.
        """
        
        for num_features in range(1, max_dim+1):
            feature_combo_list = list(combinations(features, num_features))        
            for feature_subset in feature_combo_list:
                f_search(k_range, feature_subset)
                
    #-------------------------------------------
    def k_search(s, k_range, features):
        """ Runs kmeans, saves cluster file, saves plot file,
            returns silhoutte and Calinski scores 
        """
        logging.info('starting cluster search' )
        logging.info('\tfeatures=' + str(features))

        cluster_list = []
        sil_scores = []
        ch_scores = []
        X = s.df.loc[:,features].dropna().values
        logging.info('\tX.shape' + str(X.shape))
        score_fname = 'output/scores_' + \
            '_'.join(features).replace(' ','')+ '.csv'
        for k in k_range:
            png_fname = 'output/clusters_' + '_'.join(features).replace(' ','')\
                + '_' + str(k) + '.png'
            cluster_fname = 'output/face_clusters_' + \
                '_'.join(features).replace(' ','') + '_' + str(k) + '.csv'
            if os.path.isfile(png_fname):
                logging.info('file exists, skipping...' + png_fname)
                continue
            logging.info('\tk=' + str(k) )
            k_means = cluster.KMeans(n_clusters=k, max_iter=1000, n_jobs=-1)
            k_means.fit(X)
            y = k_means.predict(X)
            
            sil_score = s.calc_sil_score(X,y)
            ch_score = calinski_harabaz_score(X,y)
            
            logging.info('silhouette score with ' + str(k) + ' clusters: ' + \
                             '{0:.3f}'.format(sil_score))
            logging.info('CH score with ' + str(k) + ' clusters: ' + \
                             '{0:.3f}'.format(ch_score))
            
            clusters = k_means.cluster_centers_
                
            # write the clusters to a csv file
            s.write_clusters(cluster_fname, features, clusters)
            subtitle = 'sil score: ' + '{:.3f}'.format(sil_score)
            subtitle += ', CH score: ' + '{:.2E}'.format(ch_score)
            s.write_plots(png_fname, features, X,[clusters], subtitle)  
            s.write_scores(score_fname, features, [k], [sil_score], [ch_score])

            cluster_list.append(clusters)
            sil_scores.append(sil_score)
            ch_scores.append(ch_score)
        
        if len(cluster_list) > 1:
            k_start = cluster_list[0].shape[0]
            k_end = cluster_list[-1].shape[0]
            fname = 'output/clusters_' + '_'.join(features).replace(' ','') + '_R' + \
                                    str(k_start) + 'to' + str(k_end) + '.png'
            s.save_score_plot(score_fname.replace('.csv','.png'), k_range, ch_scores)
            
        
        return sil_scores, ch_scores
    
    #-------------------------------------------
    def gmm_search(s, k_range, features, origin_cluster=False):
        """ Runs GMM, saves medioid file, saves plot file.
            returns silhoutte and Calinski scores 
        """
        log.info('starting cluster search' )
        log.info('\tfeatures=' + str(features))

        bic_scores = []
        X = s.df.loc[:,features].dropna().values
        if origin_cluster:
            X_ = X[~(X == 0).all(1)] # drop any rows with all zero
        else:
            X_ = X
        log.info('\tX.shape' + str(X.shape))
        for k in k_range:
            png_fname = 'output/gmm_' + '_'.join(features).replace(' ','')\
                + '_' + str(k) + '.png'
            if os.path.isfile(png_fname):
                log.info('file exists, skipping...' + png_fname)
                continue
            log.info('\tk=' + str(k) )
            #k_means = cluster.KMeans(n_clusters=k, max_iter=1000, n_jobs=-1)
            gmm = mixture.GaussianMixture(n_components=k,
                                          covariance_type='full',
                                          #tol=1e-8,
                                          tol=1e-6,
                                          max_iter=1000,
                                          #max_iter=200,
                                          n_init=3,
                                          reg_covar=2e-6)            
            gmm.fit(X_)
            bic = gmm.bic(X_)  
            log.info('bic with ' + str(k) + ' clusters: ' + \
                             '{0:.3f}'.format(bic))
                
            # write the clusters to a csv file
            output_file = 'output/face_clusters_' + \
                '_'.join(features).replace(' ','') + '_' + str(k) + '.csv'
            outfile = 'output/gmm_' 
            outfile += '_'.join(features).replace(' ','') + '_' + str(k) + '.csv'            
            #s.write_gmm_clusters(outfile, gmm, features, clusters)
            subtitle = ', BIC: ' + '{:.2E}'.format(bic)
            if len(features) <= 2:
                s.write_gmm_plots(png_fname, gmm, features, X, origin_cluster, subtitle)
            s.write_bics(features, [k], [bic])
            bic_scores.append(bic)
        
        return bic_scores
    
    
    #-------------------------------------------
    def bin_search(s, k_range, features):
        """ Counts all possible binary combinations of features in data. 
        """
        log.info('starting bin search' )
        log.info('\tfeatures=' + str(features))

        cluster_list = []
        sil_scores = []
        ch_scores = []
        X = s.df.loc[:,features].dropna().values
        log.info('\tX.shape' + str(X.shape))
        count_dict = {}
        for i in list(itertools.product([0, 1], repeat=len(features))):    
            count_dict[i] = 0
            
        count_tensor = np.zeros(len(features), dtype=int)
        for i in range(X.shape[0]):
            #count_tensor[X.iloc[i]]] += 1
            count_dict[tuple(X[i])] += 1
        
        sorted_counts = sorted(count_dict.items(), key=operator.itemgetter(1))
        total = sum(count_dict.values())
        print(features)
        for v,k in sorted_counts:
            print(v ,  ' {:.3}'.format(k/total))
            
        return    
    
    #-------------------------------------------
    def write_clusters(s, outfile, features, clusters):
        """ Writes cluster means to a csv file. 
        """
        with open(outfile, 'w') as f:
            feature_data = []
            writer = csv.writer(f)
            writer.writerow(features)
        
            for cluster_i in clusters:
                row_txt = [str(x) for x in list(cluster_i)]
                writer.writerow(row_txt)        
    
    
    #-------------------------------------------
    def write_gmm_clusters(s, outfile, clf, features, X,  subtitle=None):
        """ Saves the means, covariances, and weights of gmm clusters in clf. 
            TODO: add weights (priors).
        """

        print(clf.means_)
        sigmas = np.empty(clf.covariances_.shape[0],dtype=object)
        for i in range(sigmas.shape[0]):
            sigmas[i] = clf.covariances_[i]
        cluster_data = np.concatenate((clf.means_,sigmas[:,np.newaxis]),axis=1)
        df_clusters = pd.DataFrame(data=cluster_data,columns=features+['sigmas'])
        df_clusters.to_csv(outfile,index=False)
    
    #-------------------------------------------
    def write_gmm_plots(s, fname, clf, features, X,  origin_cluster, 
                        subtitle=None):
        """ Saves 1d and 2d plots of gmm specified in clf.
            TODO: plot tsne when d > 2.
        """
        
        plt.figure(figsize=(8,8))
        k,d = clf.means_.shape
        if origin_cluster:
            X_ = X[~(X == 0).all(1)] # drop any rows with all zero
        else:
            X_ = X
        Y_ = clf.predict(X_)
        color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                      'darkorange'])

        plt.title('GMM: k=' + str(k) + ', ' + ','.join(features) + ' ' \
                  + subtitle)
        plt.xlabel(features[0])
        
        if d==2:
            for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                                       color_iter)):
                plt.ylabel(features[1])
                v, w = linalg.eigh(cov)
                if not np.any(Y_ == i):
                    continue
                plt.scatter(X_[Y_ == i, 0], X[Y_ == i, 1], .8, color=color,alpha=.7)
    
                x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 50)
                y_vals = np.linspace(X[:,1].min(), X[:,1].max(), 50)
                x, y = np.meshgrid(x_vals, y_vals)    
                pos = np.empty(x.shape + (2,))
                pos[:, :, 0] = x; pos[:, :, 1] = y
                rv = multivariate_normal(mean, cov)
            
                try: # not sure why, was running into ValueErrors
                    plt.contour(x, y, rv.pdf(pos))
                except ValueError:
                    pass
                
        elif d==1:
            rv_sum = np.zeros(100)
            
            x = np.linspace(min(X),max(X),100)
            bins = np.linspace(min(X),max(X),100)
            plt.hist(X[:,0], bins=bins, color='green', histtype='bar', \
                     ec='black', normed=True)
            axes = plt.gca()
            ylim = axes.get_ylim()
            for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                                               color_iter)):
                rv = multivariate_normal(mean, cov)
                plt.plot(x, clf.weights_[i]*rv.pdf(x), lw=2)
                rv_sum += clf.weights_[i]*rv.pdf(x)

            plt.plot(x, rv_sum, lw=5, color='red', alpha=.4)           
            axes.set_ylim(ylim)

        plt.savefig(fname)
        plt.close()
        
    #-------------------------------------------
    def calc_sil_score(s,X,y):
        """ Calculate silhoutte score with efficiency mods.
        """
        if(X.shape[0] > 5000):
            # due to efficiency reasons, we need to only use subsample
            sil_score_list = []
            for i in range (0,100):
                X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                     test_size=2000./X.shape[0])        
                sil_score = silhouette_score(X_test,y_test)
                sil_score_list.append(sil_score)
            sil_score_avg = np.nanmean(sil_score_list)
        else:   
            sil_score_avg = silhouette_score(X,y)

        return sil_score_avg
        
        
    #-------------------------------------------
    def save_score_plot(s, score_fname, k_range, ch_scores):
        """ Save png files for plots of data with clusters.
            TODO: plot tsne when d > 2.
        """
        plt.plot(k_range, ch_scores)
        plt.xlabel('# clusters')
        plt.ylabel('CH score')
        plt.title('CH scores ' + score_fname)
        plt.savefig(score_fname)
        plt.close()
        
        
    #-------------------------------------------
    def write_plots(s, fname, header, data, cluster_list,  subtitle=None):
        """ Save png files for plots of data with clusters.
            TODO: plot tsne when d > 2.
        """
        num_clusters = len(cluster_list)
        if num_clusters == 0:
            return
        num_col = math.ceil(num_clusters/2.)
        num_row = math.ceil(num_clusters/float(num_col))
        plt.figure(figsize=(8,8))
        
        for i,clusters in enumerate(cluster_list):
            k = clusters.shape[0]
            d = clusters.shape[1]
            if d > 2:
                continue
            myTitle = 'k=' + str(k) + ' '
            if subtitle:
                myTitle += ', ' + subtitle

            if d==2:
                if len(cluster_list) > 1:
                    plt.subplot(num_row, num_col,i+1)                                
                    plt.scatter(data[:,0], data[:,1], alpha = 0.01)
                    plt.scatter(clusters[:,0], clusters[:,1], s=500, \
                                c='red', alpha =0.5)
                    plt.xlabel(header[0])
                    plt.ylabel(header[1])
                else:
                    nullfmt = NullFormatter()         # no labels
                    
                    # definitions for the axes
                    left, width = 0.1, 0.65
                    bottom, height = 0.1, 0.65
                    bottom_h = left_h = left + width + 0.05
                    
                    rect_scatter = [left, bottom, width, height]
                    rect_histx = [left, bottom_h, width, 0.2]
                    rect_histy = [left_h, bottom, 0.2, height]
                    
                    axScatter = plt.axes(rect_scatter)
                    axHistx = plt.axes(rect_histx)
                    axHisty = plt.axes(rect_histy)
                    
                    # no labels
                    axHistx.xaxis.set_major_formatter(nullfmt)
                    axHisty.yaxis.set_major_formatter(nullfmt)
                    
                    # the scatter plot:
                    axScatter.scatter(data[:,0], data[:,1], alpha = 0.01)
                    axScatter.scatter(clusters[:,0], clusters[:,1], s=500, \
                                      c='red', alpha =0.5)
                    
                    # now determine nice limits by hand:
                    binwidth = 0.1
                    xymax = np.max([np.max(np.fabs(data[:,0])), \
                                    np.max(np.fabs(data[:,1]))])
                    lim = (int(xymax/binwidth) + 1) * binwidth
                    
                    axScatter.set_xlim((0, lim))
                    axScatter.set_ylim((0, lim))
                    
                    bins = np.arange(0, lim + binwidth, binwidth)
                    axHistx.hist(data[:,0], bins=bins, color='green', \
                                 histtype='bar', ec='black', normed=1)
                    axHisty.hist(data[:,1], bins=bins, orientation='horizontal',\
                                 histtype='bar', ec='black' , normed=1)
                    
                    axHistx.set_xlim(axScatter.get_xlim())
                    axHisty.set_ylim(axScatter.get_ylim())                
                    axScatter.set_title(myTitle)
                    axScatter.set_xlabel(header[0])
                    axScatter.set_ylabel(header[1])
                
            else:
                if len(cluster_list) > 1:
                    plt.subplot(num_row, num_col,i+1)        
                plt.hist(data[:,0], 50, color='green', \
                     histtype='bar', ec='black', normed=1)                    
                #plt.scatter(data[:,0], np.ones(data.shape[0]), alpha = 0.01)
                plt.scatter(clusters[:,0], np.zeros(clusters.shape[0]), s=500, \
                            c='red')
                plt.xlabel(header[0])
                    
                plt.title(myTitle)
                #plt.tight_layout()            
        plt.savefig(fname)
        plt.close()
            
    #-------------------------------------------
    def write_scores(s, score_fname, features, k_range, sil_scores, ch_scores):
        df_scores= pd.DataFrame.from_items([
            ('features', '_'.join(features).replace(' ','')),
            ('sil score', sil_scores),
            ('CH score', ch_scores)])
        df_scores.index = k_range[0:len(sil_scores)]
        # append to score file if exists, otherwise create new
        if os.path.isfile(score_fname):
            with open(score_fname, 'a') as f:
                df_scores.to_csv(f, header=False,index_label='k')        
        else:
            df_scores.to_csv(score_fname, index_label='k')             

    #-------------------------------------------
    def write_bics(s, features, k_range, bics):
        df_scores= pd.DataFrame.from_items([
            ('features', '_'.join(features).replace(' ','')),
            ('bic', bics)])
        df_scores.index = k_range[0:len(bics)]
        # append to score file if exists, otherwise create new
        if os.path.isfile('output/bic_scores.csv'):
            with open('output/bic_scores.csv', 'a') as f:
                df_scores.to_csv(f, header=False,index_label='k')        
        else:
            df_scores.to_csv('output/bic_scores.csv', index_label='k') 

#------------------------------------------------------------------------
def do_all(args):
    if not os.path.isdir('output'):
        os.mkdir('output')

    c_search = ClusterSearch(args.i)    
    
    features=['AU12_r','AU14_r']
    # all features:
    #features=['AU01_r','AU02_r','AU04_r', 'AU05_r', 'AU06_r','AU07_r','AU09_r',\
    #          'AU10_r','AU12_r','AU14_r','AU15_r','AU17_r','AU20_r',\
    #          'AU23_r','AU25_r','AU26_r','AU45_r']
    # smile features:
    #features=['AU06_r','AU07_r',\
    #          'AU10_r','AU12_r','AU14_r']
    max_d = 3

    if args.t == 'km':
        c_search.feature_search(features, range(2,args.k+1), max_d, c_search.k_search)
    elif args.t == 'gmm':
        c_search.feature_search(features, range(1,args.k+1), max_d, c_search.gmm_search)
    elif args.t == 'bin':
        features=['AU01_c','AU02_c','AU04_c','AU05_c', 'AU06_c','AU07_c',\
                  'AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c',\
                  'AU20_c','AU23_c','AU25_c','AU26_c','AU45_c']
        c_search.feature_search(features, range(1,args.k+1), 3, c_search.bin_search)        
    
#------------------------------------------------------------------------
if __name__ == '__main__':

    # Setup commandline parser
    help_intro = 'Program for running clustering on features subsets.'
    help_intro += '  example usage:\n\t$ ./cluster.py -i \'example/test.csv\''
    help_intro += ' -t gmm -k 6'
    parser = argparse.ArgumentParser(description=help_intro)

    parser.add_argument('-i', help='inputs, ex:example/test.csv', type=str, 
                        default='example/test.csv')
    parser.add_argument('-t', help='type: gmm, km', type=str, 
                        default='km')
    parser.add_argument('-k', help='maximum k', type=int, 
                        default=14)
    
    args = parser.parse_args()
    
    print('args: ', args.i)

    do_all(args)
    log.info('PROGRAM COMPLETE')

