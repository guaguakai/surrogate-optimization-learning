#!/bin/bash
#SBATCH -n 4                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 2-00:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=20000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_variablephi_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/5.0.1-fasrc02 cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01
source activate aditya

cd ~/projects/smuggler_networks
sh singletest_part4.sh 
