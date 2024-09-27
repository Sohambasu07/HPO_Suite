#!/bin/bash
#SBATCH --partition mlhiwidlc_gpu-rtx2080    # short: -p mlhiwidlc_gpu-rtx2080
#SBATCH --job-name HPOSuite_test_SLURMArray            #  short: -J HPOSuite_test_SLURMArray
#SBATCH --output logs/%x_%A_%a.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x_%A_%a.out
#SBATCH --error logs/%x_%A_%a.err    # STDERR  short: -e logs/%x_%A_%a.err
#SBATCH --array=1-20

# Sample SLURM script to run array job from a dump file grouped by memory requirements
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

# # Count the number of lines in the dump file
# TOTAL_LINES=$(wc -l < $dump_file_path)

# # Get the specific command from the file for this array task
# COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $COMMAND_FILE)

# # Print the command (for debugging purposes)
# echo "Running command: $COMMAND"

# # Execute the command
# eval $COMMAND

# Read the file line by line and execute each command as a shell command
while IFS= read -r line
do
  echo "Executing: $line"
  $line  # Run the Python command as a shell command
done < "$dump_file_path"

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
