#!/bin/bash
#SBATCH --partition mlhiwidlc_gpu-rtx2080    # short: -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name HPOSuite_test            #  short: -J HPOSuite_test
#SBATCH --output logs/%x_%A.out   # STDOUT  short: -o logs/%x_%A.out
#SBATCH --error logs/%x_%A.err    # STDERR  short: -e logs/%x_%A.err

# Sample SLURM script to run jobs from a dump file
# NOTE: $dump_file_path is a placeholder for the path to the dump file and must be replaced with the actual path to the dump file in the SBATCH command.


# Check if the dump file exists
if [ ! -f $dump_file_path ]; then
    echo "Dump file not found."
    exit 1
fi

if [[ $dump_file_path =~ dump_([0-9]+)MB\.txt$ ]]; then
    MEMORY="${BASH_REMATCH[1]}MB"  # Extract memory (e.g., 2048MB)
else
    MEMORY="4096MB"  # Default value in MB if no match found
fi

# Set the memory limit
#SBATCH --mem=$MEMORY
echo "Memory limit set to $MEMORY"

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
source ~/repos/automl_env/bin/activate

start=`date +%s`

while IFS= read -r line
do
  echo "Executing: $line"
  $line
done < "$dump_file_path"

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
