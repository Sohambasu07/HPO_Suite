#!/bin/bash
#SBATCH --partition mlhiwidlc_gpu-rtx2080    # short: -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name HPOSuite_dump           #  short: -J HPOSuite_dump
#SBATCH --output logs/%x_%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x_%A_job_name.out
#SBATCH --error logs/%x_%A.err    # STDERR  short: -e logs/%x_%A_job_name.err
#SBATCH --mem=2GB 

# Sample SLURM script to run experiments from a configuration file
# $CONFIG_FILE is a placeholder for the path to the configuration file and must be replaced with the actual path to the configuration file in the SBATCH command.

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
source ~/repos/automl_env/bin/activate

start=`date +%s`

# Activate your environment
source ~/repos/automl_env/bin/activate

# Command to run from config file
python -m hpo_glue --exp_config $CONFIG_FILE

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
