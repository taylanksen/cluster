#!/bin/bash
#SBATCH --partition=standard --time=02:00:00
#SBATCH -c 24

# Takes cluster definition file and input data folder and generates cluster index sequences 
# for each type (default|listening|speaking|changes) at various sample rates (1,3,5,10,15,20)
# using converter.py with varying arguments.

# Usage: 
#     python batch_convert.sh cluster_defintion input_data_folder
#         To run with custom input on interactive node.
#     sbatch batch_convert.sh 
#         To run with default params on a compute node. 

# (defaults to the same defaults as converter.py if no arguments given)

module load python
module load anaconda

# Permutations to run
sample_rates=( 1 3 5 10 15 20 ) # Sample Rates
types=( default listening speaking changes ) # Folders of input sequences

# Default argument values
cluster_def='/public/tsen/clusterResults/km_all_k15_d2/face_clusters_AU06_r_AU12_r_5.csv'
data_folder='/public/tsen/OpenFace'

# Parse CL args if given -- Either both or none can be given, not just one
if [[ $# == 2 ]]; then
	cluster_def=$1
	data_folder=$2
fi

# # Confirm arguments -- (y/n) to continue or cancel
# printf "Continue with:\n\tdata_folder = $data_folder\n\tcluster_def = $cluster_def\n\nContinue? (y/n) "
# read decision
decision="y"

if [[ "$decision" == "y" || "$decision" == "Y" ]]; then
	echo "Beginning conversions to cluster index sequences..."
	for type in ${types[@]}; do	
		if [[ $type == "changes" ]]; then # Changes doesn't try multiple sample rates
			echo "python converter.py -t $type -s 1 -d $data_folder -c $cluster_def"
			python converter.py -t "$type" -s 1 -d "$data_folder" -c "$cluster_def"
		else
			for sample_rate in ${sample_rates[@]}; do # The other types use multiple sample rates
				echo "python converter.py -t $type -s $sample_rate -d $data_folder -c $cluster_def"
				python converter.py -t "$type" -s "$sample_rate" -d "$data_folder" -c "$cluster_def"
			done
		fi
	done
else
	echo "Cancelling batch_convert.sh..."
fi


printf "\nbatch_convert.sh complete\n"