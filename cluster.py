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
import csv
import numpy as np
import pandas as pd
import math

#import compare
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import glob
import os
import sys
import argparse
import logging

from collections import defaultdict
from itertools import combinations

logging.basicConfig(level=logging.DEBUG)
# you can set the logging level from a command-line option such as:
#   --log=INFO


#-------------------------------------------------------------------------------
class KmeansSearch():
    """ """
    
    def __init__(s, infile):
        s.df = pd.read_csv(infile) 
        
        
    #-------------------------------------------
    def feature_search(s, features, k_range, max_dim):
        """ runs k_search for all permutations of features
        """
        
        for num_features in range(1, max_dim+1):
            feature_combo_list = list(combinations(features,num_features))        
            for feature_subset in feature_combo_list:
                s.k_search(k_range, feature_subset)
                
                
    #-------------------------------------------
    def k_search(s, k_range, features):
        """ runs kmeans, saves cluster file, saves plot file
            returns silhoutte and Calinski scores 
        """
        logging.info('starting cluster search' )
        logging.info('\tfeatures=' + str(features))

        cluster_list = []
        sil_scores = []
        ch_scores = []
        X = s.df.loc[:,features].dropna().values
        logging.info('\tX.shape' + str(X.shape))
        for k in k_range:
            png_fname = 'output/clusters_' + '_'.join(features).replace(' ','')\
                + '_' + str(k) + '.png'
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
            output_file = 'output/face_clusters_' + \
                '_'.join(features).replace(' ','') + '_' + str(k) + '.csv'
            s.write_clusters(output_file, features, clusters)
            subtitle = 'sil score: ' + '{:.3f}'.format(sil_score)
            subtitle += ', CH score: ' + '{:.2E}'.format(ch_score)
            s.write_plots(features, X,[clusters], subtitle)  
            s.write_scores(features, [k], [sil_score], [ch_score])

            cluster_list.append(clusters)
            sil_scores.append(sil_score)
            ch_scores.append(ch_score)

        s.write_plots(features, X, cluster_list)
        
        return sil_scores, ch_scores
    
    #-------------------------------------------
    def write_clusters(s, outfile, features, clusters):
        with open(outfile, 'w') as f:
            feature_data = []
            writer = csv.writer(f)
            writer.writerow(features)
        
            for cluster_i in clusters:
                row_txt = [str(x) for x in list(cluster_i)]
                writer.writerow(row_txt)        
    
    #-------------------------------------------
    def calc_sil_score(s,X,y):
        """ Calculate silhoutte score with efficiency mods """
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
    def write_plots(s, header, data, cluster_list,  subtitle=None):
        num_clusters = len(cluster_list)
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
                    
                    # start with a rectangular Figure
                    #plt.figure(1, figsize=(8, 8))
                    
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
                plt.hist(data[:,0], 25, color='green', \
                     histtype='bar', ec='black', normed=1)                    
                #plt.scatter(data[:,0], np.ones(data.shape[0]), alpha = 0.01)
                plt.scatter(clusters[:,0], np.zeros(clusters.shape[0]), s=500, \
                            c='red')
                plt.xlabel(header[0])
                    
                plt.title(myTitle)
        if len(cluster_list) > 1:
            plt.tight_layout()      
        #plt.show()
        if(len(cluster_list) == 1):
            plt.savefig('output/clusters_' + '_'.join(header).replace(' ','') + '_' \
                        + str(k) + '.png')
        else:
            k_start = cluster_list[0].shape[0]
            k_end = cluster_list[-1].shape[0]
            plt.savefig('output/clusters_' + '_'.join(header).replace(' ','') + '_R' + \
                        str(k_start) + 'to' + str(k_end) + '.png')
            plt.close()
            
    #-------------------------------------------
    def write_scores(s, features, k_range, sil_scores, ch_scores):
        df_scores= pd.DataFrame.from_items([
            ('features', '_'.join(features).replace(' ','')),
            ('sil score', sil_scores),
            ('CH score', ch_scores)])
        df_scores.index = k_range[0:len(sil_scores)]
        # append to score file if exists, otherwise create new
        if os.path.isfile('output/scores.csv'):
            with open('output/scores.csv', 'a') as f:
                df_scores.to_csv(f, header=False,index_label='k')        
        else:
            df_scores.to_csv('output/scores.csv', index_label='k')        
    
#------------------------------------------------------------------------
def do_all(args):
    kmeans_search = KmeansSearch(args.i)    
    
    features=[' AU06_r',' AU12_r']
    #features=[' AU01_r',' AU02_r',' AU05_r', ' AU06_r',' AU07_r',' AU09_r',\
    #          ' AU10_r',' AU12_r',' AU14_r',' AU15_r',' AU17_r',' AU20_r',\
    #          ' AU23_r',' AU25_r',' AU26_r',' AU45_r']
    if not os.path.isdir('output'):
        os.mkdir('output')
    k_range = range(2,5)
    max_d = 2
    #sil_score, ch_score = kmeans_search.k_search(k_range,features)
    kmeans_search.feature_search(features, k_range, max_d)
    
#------------------------------------------------------------------------
if __name__ == '__main__':

    # Setup commandline parser
    help_intro = 'Program for running kmeans variants.' 
    #help_intro += ' example usage:\n\t$ ./avg_master.py -i \'example/*_openface.txt\''
    parser = argparse.ArgumentParser(description=help_intro)

    parser.add_argument('-i', help='inputs, ex:example/*.txt', type=str, 
                        default='example/test.csv')
    args = parser.parse_args()
    
    print('args: ',args.i)

    do_all(args)
    logging.info('PROGRAM COMPLETE')

