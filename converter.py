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

import matplotlib.pyplot as plt

import glob
import os
import sys
import argparse
import logging


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
    
#------------------------------------------------------------------------
def do_all(args):
    cluster_converter = ClusterConverter()    
    cluster_converter.load_cluster(args.c)
    cluster_converter.load_data_sequence(args.d)
    cluster_converter.write_cluster_sequence(args.o)

#------------------------------------------------------------------------
if __name__ == '__main__':

    # Setup commandline parser
    help_intro = 'Program for converting a seq datafile into a seq of cluster indices.' 
    parser = argparse.ArgumentParser(description=help_intro)

    parser.add_argument('-c',help='cluster_file, ex:example/face_clusters.csv',\
                        type=str, default='example/face_clusters_AU06_r_AU12_r_4.csv')
    parser.add_argument('-d', help='data_file, ex:example/test.csv', \
                        type=str, default='example/test.csv')
    parser.add_argument('-o', help='output_filename, ex:output/test.seq', \
                        type=str, default='output/test.seq')
    args = parser.parse_args()
    
    print('args: ', args)

    do_all(args)
    logging.info('PROGRAM COMPLETE')

