#!/bin/bash

# Run python through the grid engine

# Specifies the job's name, which controls the name of the output files.
# Output files will be of the form <JOB NAME>.o<JOBID>, for example.
#$ -N precip_stats

# Specifies the queue to wait in. The queue determines which cluster nodes
# your job will be run on. We only have the all.q.
#$ -q all.q

# Specify the parallel environment to run in.
#$ -pe mpi 20

# Mail options
#$ -m beas
#$ -M jcdehart@rams.colostate.edu

# Redirect stdout and stderr (these are directories)
#$ -o /bell-scratch/jcdehart/precip/rotation/pyscripts/job_out
#$ -e /bell-scratch/jcdehart/precip/rotation/pyscripts/job_err

# ---END OF GRID ENGINE OPTIONS---

# When grid engine starts your job, the current directory will be your
# home directory. You may want to go to another directory. Uncomment the
# line below and edit the directory name.

cd /bell-scratch/islushy/JMLQC/JMLQC
# The actual command to run

export OMP_NUM_THREADS=20
#python -W ignore -u precip_meiyu_stats_vort_w.py > maui_stats_mp_vortw.log
julia JL_BENCHMARK_SCRIPT.jl > JL_Benchmark.out 
