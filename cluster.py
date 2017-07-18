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

import glob
import os
import sys
import argparse
import logging

from collections import defaultdict

logging.basicConfig(level=logging.DEBUG)
'''
If you want to set the logging level from a command-line option such as:
  --log=INFO
'''

#-------------------------------------------------------------------------------
class KmeansSearch():
    """ """
    
    def __init__(s, infile):
        s.df = pd.read_csv(infile) 
        
    #-------------------------------------------
    def cluster(s, k_range, features):
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
            logging.info('\tk=' + str(k) )
            k_means = cluster.KMeans(n_clusters=k, max_iter=1000, n_jobs=1)
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
            output_file = 'face_clusters_' + str(k) + '.csv'
            s.write_clusters(output_file, features, clusters)
            subtitle = 'sil score: ' + '{0:.3f}'.format(sil_score)
            s.plot_cluster_and_data(features, X,[clusters], subtitle)  

            cluster_list.append(clusters)
            sil_scores.append(sil_score)
            ch_scores.append(ch_score)
            s.plot_cluster_and_data(features, X, cluster_list)
        s.write_k_search(k_range, sil_scores, ch_scores)
        
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
    def plot_cluster_and_data(s, header, data, cluster_list,  subtitle=None):
        num_clusters = len(cluster_list)
        num_col = math.ceil(num_clusters/2.)
        num_row = math.ceil(num_clusters/float(num_col))
        plt.figure(figsize=(8,8))
        
        for i,clusters in enumerate(cluster_list):
            k = clusters.shape[0]
            myTitle = 'k=' + str(k) + ' '
            if subtitle:
                myTitle += ', ' + subtitle
            plt.subplot(num_row, num_col,i+1)
            plt.scatter(data[:,0], data[:,1], alpha = 0.01)
            plt.scatter(clusters[:,0], clusters[:,1], s=500, c='red', alpha =0.5)
            plt.xlabel(header[0])
            plt.ylabel(header[1])
            plt.title(myTitle)
        plt.tight_layout()      
        #plt.show()
        if(len(cluster_list) == 1):
            plt.savefig('clusters_' + str(k) + '.png')
        else:
            k_start = cluster_list[0].shape[0]
            k_end = cluster_list[-1].shape[0]
            plt.savefig('clusters_' + str(k_start) + 'to' + str(k_end) + '.png')
            

    def write_k_search(s, k_range, sil_scores, ch_scores):
        df_scores= pd.DataFrame.from_items([('sil score', sil_scores),\
                                            ('CH score', ch_scores)])
        df_scores.index = k_range
        df_scores.to_csv('scores.csv',index_label='k')
            
#------------------------------------------------------------------------
def do_all(args):
    kmeans_search = KmeansSearch(args.i)    
    
    features=[' AU06_r',' AU12_r']
    sil_score, ch_score = kmeans_search.cluster(range(2,5),features)
    
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