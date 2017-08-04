#!/usr/bin/env python
"""
------------------------------------------------------------------------

------------------------------------------------------------------------
"""
from __future__ import print_function
import csv
import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn import mixture
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

import glob
import os
import sys
import argparse
import logging
import re

logging.basicConfig(level=logging.DEBUG)
# you can set the logging level from a command-line option such as:
#   --log=INFO


#------------------------------------------------------------------------
class ClusterConverter:
    """ """
    def load_cluster(s, infile):
        """ loads a cluster file into a df """
        print("Reading cluster definition from: {}".format(infile))
        df = pd.read_csv(infile, skipinitialspace=True)
        s.cluster_def = df # Cluster definition as a dataframe
        print("Cluster Centers: \n", df)
    
    def load_data_sequence(s, infile):
        """ loads sequence of selected features into a df """
        print('Reading data sequence from: {}'.format(infile))
        # Read the infile for only the features designated by cluster_def
        df = pd.read_csv(infile, skipinitialspace=True, usecols=list(s.cluster_def))
        s.data = df # Dataframe for the data points
    
    def write_cluster_sequence(s,outfile):
        """ uses KMeans prediction to print sequence of closest clusters """
        k = s.cluster_def.shape[0] # k is the number of clusters
        k_means = cluster.KMeans(n_clusters=k, max_iter=2000, n_init=10)
        k_means.cluster_centers_ = s.cluster_def # Set the cluster centers
        
        np.set_printoptions(threshold=np.inf) # So it prints whole array, w/o ...
        
        # Predict the closest cluster for the sequence entries and write to file
        with open(outfile, 'w') as f:
            f.write(str(k_means.predict(s.data))[1:-1]) #[1:-1] to remove the '[]'

        print('Cluster Index Sequnce written to: {}'.format(outfile))
        
    """ """
def load_gmm_cluster(infile):
    """ loads a gmm cluster file into a df """
    print("Reading cluster definition from: {}".format(infile))
    
    means = []
    sigmas = []
    
    df = pd.read_csv(infile)
    values = df.values
    
    for i in range(0, values.shape[0]):
        current_means = values[i].reshape(values[i].shape[0],1)[:-1]
        current_sigmas = values[i].reshape(values[i].shape[0],1)[-1]
        temp = current_sigmas[0].split("\n")
        
        row_list =[]
        for j in range(0, len(temp)):
            row = re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", temp[j])
            y = (np.array(row)).astype(np.float)
            y = y.reshape(1, len(temp))
            row_list.append(y)
        
        current_sigmas = row_list[0]
        for j in range(1, len(row_list)):
            current_sigmas = np.concatenate((current_sigmas, row_list[j]), axis=0)
            
               
        means.append(current_means)
        sigmas.append(current_sigmas)
    
    means = np.array(means)
    sigmas = np.array(sigmas)
    print("Cluster centers:\n", means)
    print("Cluster covariance:\n", sigmas)
    
    return means,sigmas

def write_gmm_cluster_sequence(means, sigmas, outfile, targetFile):
    """ uses GMM prediction to print sequence of closest clusters """
    k = means.shape[0]
    features=[' AU06_r',' AU12_r']
    
    df = pd.read_csv(targetFile)
    data = df.loc[:,features].dropna().values
    
    probability_list = []
    output = []
    means = means.reshape(means.shape[0],means.shape[1])
    for i in range(0,k):
        y = multivariate_normal.pdf(data, means[i], sigmas[i])
        probability_list.append(y)
        
    for i in range(0, probability_list[0].shape[0]):
        currentMax = 0
        for j in range(0,len(probability_list)):
            if(probability_list[j][i]>probability_list[currentMax][i]):
                currentMax = j
        output.append(currentMax)
    output = np.array(output)

    
    with open(outfile, 'w') as f:
        for i in range(0, output.shape[0]):
            f.write(str(output[i])+' ')
        
    print('Cluster Index Sequnce written to: {}'.format(outfile))
            
#------------------------------------------------------------------------
def do_all(args):
    cluster_converter = ClusterConverter()    
    #cluster_converter.load_cluster(args.c)
    #cluster_converter.load_data_sequence(args.d)
    #cluster_converter.write_cluster_sequence(args.o)
    means, sigmas = load_gmm_cluster(args.c)
    write_gmm_cluster_sequence(means, sigmas, args.o, args.t)

#------------------------------------------------------------------------
if __name__ == '__main__':

    # Setup commandline parser
    help_intro = 'Program for converting a seq datafile into a seq of cluster indices.' 
    parser = argparse.ArgumentParser(description=help_intro)

    #parser.add_argument('-c',help='cluster_file, ex:example/face_clusters.csv',\
    #                    type=str, default='example/face_clusters_AU06_r_AU12_r_4.csv')    
    parser.add_argument('-c',help='cluster_file, ex:example/face_clusters.csv',\
                        type=str, default='test_gmm_5_inp.csv')
    parser.add_argument('-d', help='data_file, ex:example/test.csv', \
                        type=str, default='example/test.csv')
    parser.add_argument('-o', help='output_filename, ex:output/test.seq', \
                        type=str, default='output/test_gmm.seq')
    parser.add_argument('-t', help='output_filename, ex:output/test.seq', \
                        type=str, default='example/targetFile.txt')    
    args = parser.parse_args()
    
    print('args: ', args)

    do_all(args)
    logging.info('PROGRAM COMPLETE')

