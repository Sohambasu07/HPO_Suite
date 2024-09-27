# Sample SLURM script to run array job from a dump file grouped by memory requirements

#!/bin/bash
#SBATCH --partition mlhiwidlc_gpu-rtx2080    # short: -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name HPOSuite_test_SLURMArray            #  short: -J HPOSuite_test_SLURMArray
#SBATCH --output logs/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x_%A_%a.err    # STDERR  short: -e logs/%x-%A-job_name.out

dump_file_path=$FILEPATH

# Check if the dump file exists
if [! -f $dump_file_path]; then
    echo "Dump file not found."
    exit 1
fi

if [[ $dump_file_path =~ dump_([0-9]+MB) ]]; then
    MEMORY="${BASH_REMATCH[1]}"  # Extract memory (e.g., 2048MB, 4GB)
else
    MEMORY="4GB"  # Default value if no match found
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
