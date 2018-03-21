#!/usr/bin/env python

"""
----------------------------------------------------------------------------------
Converts folders of raw data into cluster index sequences for each witness datafile
in the folder, based on an input cluster defintion as well as type of conversion.
Listening and speaking types utilize the dialog transcript to isolate each question
and generate a sequence for each question, as opposed to one for the whole interview.

Valid Types: - default   (one sequence for the entire interview)
             - changes   (only cluster index changes are written as the sequence)
             - listening (sequence for each question, while witness is listening)
             - speaking  (sequence for each question, while witness is speaking)
             

Usage: './converter.py -t Type -c ClusterDefinition -d DataFolder -s SampleRate'

Ex: './converter.py -t default -c km_5.csv -d OpenFace -s 5'
    Writes resulting cluster index files to 'cluster_sequences/km_5/default/every_5_frames'
    inside truthers and bluffers subfolders for each witness file in OpenFace folder.

----------------------------------------------------------------------------------
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import glob
import os
import sys
import argparse
import logging
logging.basicConfig(level=logging.INFO)


"""
Contains methods to load cluster definitions and data files and write the 
resulting cluster index sequence. Below this class, wrapper functions utilize
ClusterConverter to act on the whole dataset.
"""
class ClusterConverter:
   
    """ Loads a cluster file into a df """
    def load_cluster(s, infile):
        logging.debug('Reading cluster definition from: {}'.format(infile))
        df = pd.read_csv(infile, skipinitialspace=True)
        s.cluster_def = df # Cluster definition as a dataframe
        
        k = s.cluster_def.shape[0] # k is the number of clusters
        
        if s.method == 'KMeans':
            s.k_means = KMeans(n_clusters=k)
            s.k_means.cluster_centers_ = s.cluster_def # Set the cluster centers
        if s.method == 'GMM':
            # TODO: Parse GMM cluster definition from the df
            s.gmm = GaussianMixture(n_components=k)
            #s.gmm.means_ = 
            #s.gmm.covariances_ = 
            #s.gmm.weights_ = 
            
    
    """ Loads sequence of selected features into a df """
    def load_data_sequence(s, infile, start_frame=0, sample_rate=1):
        logging.debug('Reading data sequence from: {}'.format(infile))
        # Read the infile for only the features designated by cluster_def
        df = pd.read_csv(infile, skipinitialspace=True, usecols=list(s.cluster_def))
        s.data = df[start_frame::sample_rate] # Dataframe for the data points
        # Only samples every 'sample_rate' frames, starts at given 'start_frame'
   
    """ Uses cluster definition to print sequence of closest clusters """
    def write_cluster_sequence(s, outfile):
        if s.method == 'KMeans':
            seq = s.k_means.predict(s.data) # Run k_means.predict to find closest clusters
        if s.method == 'GMM':
            seq = s.gmm.predict(s.data) # Run gmm.predict to find closest clusters   
            
        np.set_printoptions(threshold=np.inf) # So it prints whole array, not '...'
        # Predict the closest cluster for the sequence entries and write to file
        with open(outfile, 'w+') as f:
            f.write(str(seq)[1:-1]) #[1:-1] to remove the '[' and ']'

        logging.debug('Cluster Index Sequnce written to: {}'.format(outfile))

    """ Converts the passed frames from s.data based on s.k_means """
    def convert_seq(s, frames, method='KMeans'):
        
        np.set_printoptions(threshold=np.inf) # So it prints whole array, w/o ...
        #print(frames[0], frames[1], len(s.data))	    
        frames_included = s.data.iloc[frames[0]:frames[1]]
        #print(frames_included)
        if method == 'KMeans':
            return s.k_means.predict(frames_included)
        if method == 'GMM':
            return s.gmm.predict(frames_included)

    """ Only writes when the cluster has changed """
    def write_cluster_sequence_changes(s, outfile):

        np.set_printoptions(threshold=np.inf) # So it prints whole array, w/o ...

        if s.method == 'KMeans':
            seq = s.k_means.predict(s.data) # Run k_means.predict to find closest clusters
        if s.method == 'GMM':
            seq = s.gmm.predict(s.data) # Run gmm.predict to find closest clusters
        
        # just_changes = Cluster indices only where the cluster changes
        just_changes = []
        last = -1
        for entry in seq:
            if entry != last:
                just_changes.append(entry)
                last = entry
        just_changes = np.array(just_changes)

        # Write to file
        with open(outfile, 'w+') as f:
            f.write(str(just_changes)[1:-1]) #[1:-1] to remove the '[]'

        logging.debug('Cluster Changes Sequence written to: {}'.format(outfile))
        

    """ Writes the cluster sequence for an answer to its outfile """
    def write_seq(s, seq, outfile):
        with open(outfile, 'w+') as f:
            f.write(str(seq)[1:-1]) #[1:-1] to remove the '[]'        
            
#------------------------------------------------------------------------
# ------------------ End of ClusterConverter class ----------------------
#------------------------------------------------------------------------


#------------------------------------------------------------------------
""" Converts a single sequence, not a folder of sequences ('deprecated') """
def do_all(args):
    cluster_converter = ClusterConverter()    
    cluster_converter.load_cluster(args.c)
    cluster_converter.load_data_sequence(args.d, args.s)
    cluster_converter.write_cluster_sequence(args.o)

#------------------------------------------------------------------------
"""
Converts all files with '-W-' in the name from args.d folder into outfile folders
(inside either a 'truthers' or 'bluffers' subfolder) replacing .csv with .seq
"""
def convert_all(args, outfolder):

    cluster_converter = ClusterConverter()
    if 'gmm' in outfolder.lower():
        cluster_converter.method = 'GMM'
    else:
        cluster_converter.method = 'KMeans'    
    cluster_converter.load_cluster(args.c) # Load cluster definition

    # Convert all Witness files to cluster sequences
    for name in glob.glob(args.d + '/*'):
        if '-W-' in name: # Only want Witness data
            t_name = name.split('-W')[0].split('/')[-1] # Transcript name (to find end of baseline)
            try:
                start_frame = get_start_frame(t_name) # End of baseline questions
            except IOError:
                logging.debug('No transcript found - skipping {}'.format(name))
                continue
            # Load data at given sample rate, starting at the proper start_frame to exlude baseline
            cluster_converter.load_data_sequence(name, sample_rate=args.s, start_frame=start_frame)
            if '-T-' in name: # Truth-Teller
                outfile = name.replace(args.d,outfolder+'/truthers').replace('.csv','.seq')                
            if '-B-' in name: # Bluffer
                outfile = name.replace(args.d,outfolder+'/bluffers').replace('.csv','.seq')    
           
            if args.t == 'changes': # Only writes when cluster changes
                cluster_converter.write_cluster_sequence_changes(outfile)
            else:
                cluster_converter.write_cluster_sequence(outfile)
            
#------------------------------------------------------------------------
"""
Converts each question as its own sequence using dialog transcripts and OpenFace data files
Uses args.t to determine whether to convert when witness is speaking or when they're listening
"""
def convert_with_transcripts(args, outfolder):   

    cluster_converter = ClusterConverter()
    if 'gmm' in outfolder.lower():
        cluster_converter.method = 'GMM'
    else:
        cluster_converter.method = 'KMeans'
    cluster_converter.load_cluster(args.c) # Load cluster definition

    for name in glob.glob(args.d + '/*'):
        if '-W-' in name:  # For each witness data file 
            t_name = name.split('-W')[0].split('/')[-1] # Transcript name
            # Ranges for each question/answer (depending on if inspecting while listening vs. speaking)
            try:
                ranges = get_ranges(t_name, is_listening=args.t=='listening') # (Excludes baseline phase)
            except IOError:
                logging.debug('No transcript found - skipping {}'.format(name))
                continue         
            # Data needs to be samples every frame for compatibility with transcript data
            cluster_converter.load_data_sequence(name) # sample_rate = 1, start_frame = 0 (so it works w ranges)    
            for i, frame_seq in enumerate(ranges): # For each question/answer...
                if frame_seq[1] >= frame_seq[0]: # (Rarely, they're labeled wrong, ignore these)
                    cluster_seq = cluster_converter.convert_seq(frame_seq) # Converts a sequence for this answer
                    # Write sequence to proper folder with -Qi.seq ending (i is the question number)
                    if '-T-' in name:
                        outfile = name.replace(args.d,outfolder+'/truthers').replace('.csv','-Q'+str(i)+'.seq')  
                    elif '-B-' in name:
                        outfile = name.replace(args.d,outfolder+'/bluffers').replace('.csv','-Q'+str(i)+'.seq')                                            
                    # Applies the sample rate here, since needed all data to be in sync w transcript
                    cluster_converter.write_seq(cluster_seq[::args.s], outfile) # Write the cluster sequence to outfile

#------------------------------------------------------------------------
""" 
Parses the transcript file to isolate each question, returns a list of
[start_frame, end_frame] pairs (one for each non-baseline question).
"""
def get_ranges(t_name, is_listening=True):
    answer_frames = []
    df = pd.read_csv('transcripts/' + t_name + '.csv', skipinitialspace=True)
    baseline_phase = True
    for i, row in df.iterrows():
        if 'image' in str(row[2]): # End of baseline questions
            baseline_phase = False
        if not baseline_phase: # Ignore baseline questions
            try:
                if is_listening: # For when listening
                    start_frame = get_frame(float(row[0])) - 1
                    end_frame = get_frame(float(row[1])) + 1
                else: # For when speaking
                    start_frame = get_frame(float(row[3])) - 1
                    end_frame = get_frame(float(row[4])) + 1                 
                answer_frames.append([start_frame,end_frame])
            except ValueError as er:
                print('ERROR in ',t_name,er)
    return answer_frames

#------------------------------------------------------------------------
"""
Finds the frame at the start of the first non-baseline question.
"""
def get_start_frame(t_name):
    df = pd.read_csv('transcripts/' + t_name + '.csv', skipinitialspace=True)
    for i, row in df.iterrows():
        if 'image' in str(row[2]):
            return get_frame(row[0])
    print('ERROR - no start frame found in {}'.format(t_name))


#------------------------------------------------------------------------
"""
Helper function - converts transcript's time format to frame number
Frame number from Min.Sec format (i.e. 1.52 is 1 min, 52 seconds)
"""
def get_frame(x):
    # TotalSeconds = (Decimal for seconds * 100) + (Number of Minutes * 60)
    # FrameNumber  = TotalSeconds * 15 fps
    return int((((x%1) * 100) + ((x-(x%1)) * 60)) * 15) # Math!




#------------------------------------------------------------------------
# Main Method - Parse CL args, create output folders, call converter function
#------------------------------------------------------------------------
if __name__ == '__main__':

    # Command line argument parser - can use -h or --help flags to show usage
    help_intro = 'Program for converting seq datafiles into seqs of cluster indices.' 
    parser = argparse.ArgumentParser(description=help_intro)
    # -c = ClusterDefinition: Path to cluster definition file (csv)
    parser.add_argument('-c', type=str, metavar='cluster_def', 
                help='Cluster definition file, ex:example/face_clusters_AU06_r_AU12_r_4.csv',  
                default='/public/tsen/clusterResults/km_all_k15_d2/face_clusters_AU06_r_AU12_r_5.csv')
    # -d = DataFolder: Folder containing raw data files from OpenFace
    parser.add_argument('-d', metavar='data_file', help='Input OpenFace data folder, ex:example/testData', \
                        type=str, default='/public/tsen/OpenFace')
    # -t = Type: possible options = (default | changes | listening | speaking) 
    parser.add_argument('-t', help='Conversion Type (default|changes|listening|speaking)', 
                        default='default', type=str, metavar='type')
    # -s = SampleRate: Samples every X frames of the data (raw data is 15 fps)
    parser.add_argument('-s', help='Sample Rate, data sampled every X frames', 
                        metavar='sample_rate', type=int, default=1)
    args = parser.parse_args()
    
    
    # Build outfolder path (ex: cluster_seq/KM_AU06_r_AU12_r_5/default/every_frame)
    if 'gmm' in args.c:
        outfolder = 'cluster_sequences/GMM_' 
    else:
        outfolder = 'cluster_sequences/KM_'
    outfolder += args.c.split('.')[0].split('face_clusters_')[1] + '/' + args.t
    if args.t != 'changes': # Append folder name for sample rate (unless type is changes)
        if args.s == 1:
            outfolder += '/every_frame'
        else:
            outfolder += '/every_{}_frames'.format(args.s)
    else:
        args.s = 1 # Changes only can't have a sample rate (logically doesn't make sense)
    logging.info('Outfolder = ' + outfolder)
    
    # Set up directories - make them or clean them up if they already exist
    try: os.makedirs(outfolder+'/truthers') 
    except: map(os.remove, glob.glob(outfolder+'/truthers/*'))
    try: os.makedirs(outfolder+'/bluffers')
    except: map(os.remove, glob.glob(outfolder+'/bluffers/*'))    
        
    # Call proper function to convert to cluster index sequences
    if args.t in ['speaking', 'listening']:
        convert_with_transcripts(args, outfolder)
    else:
        convert_all(args, outfolder)
        
    logging.debug('PROGRAM COMPLETE')
