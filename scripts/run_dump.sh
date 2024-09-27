#!/bin/bash
#SBATCH --partition mlhiwidlc_gpu-rtx2080    # short: -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name HPOSuite_test_SLURMArray            #  short: -J HPOSuite_test_SLURMArray
#SBATCH --output logs/%x_%A_%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x_%A_%a.out
#SBATCH --error logs/%x_%A_%a.err    # STDERR  short: -e logs/%x_%A_%a.err

# Sample SLURM script to run array job from a dump file grouped by memory requirements
# NOTE: $FILEPATH is a placeholder for the path to the dump file and must be replaced with the actual path to the dump file in the SBATCH command.


dump_file_path=$FILEPATH

# Check if the dump file exists
if [! -f $dump_file_path]; then
    echo "Dump file not found."
    exit 1
fi

if [[ $DUMP_FILENAME =~ dump_([0-9]+)MB\.txt$ ]]; then
    MEMORY="${BASH_REMATCH[1]}MB"  # Extract memory (e.g., 2048MB)
else
    MEMORY="4096MB"  # Default value in MB if no match found
fi

# Set the memory limit
#SBATCH --mem=$MEMORY

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
source ~/repos/automl_env/bin/activate

start=`date +%s`



# Count the number of lines in the dump file
TOTAL_LINES=$(wc -l < $dump_file_path)

# Check if the array task ID is valid
if [ "$SLURM_ARRAY_TASK_ID" -le "$TOTAL_LINES" ]; then

  # Get the specific command for this task ID
  COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $dump_file_path)
  
  # Print the command for debugging
  echo "Running task $SLURM_ARRAY_TASK_ID with command: $COMMAND"

  # Execute the command
  eval $COMMAND
else
  echo "No command for task ID $SLURM_ARRAY_TASK_ID"
fi

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
