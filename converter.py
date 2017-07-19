#!/usr/bin/env python
"""
------------------------------------------------------------------------

------------------------------------------------------------------------
"""
from __future__ import print_function
import csv
import numpy as np
import pandas as pd

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
        pass
    
    def load_data_sequence(s, infile):
        pass
    
    def write_cluster_sequence(s,outfile):
        pass
    
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
                        type=str, default='example/face_clusters_AU06_r_2.csv')
    parser.add_argument('-d', help='data_file, ex:example/test.csv', \
                        type=str, default='example/test.csv')
    parser.add_argument('-o', help='output_filename, ex:output/test.seq', \
                        type=str, default='output/test.seq')
    args = parser.parse_args()
    
    print('args: ', args)

    do_all(args)
    logging.info('PROGRAM COMPLETE')

