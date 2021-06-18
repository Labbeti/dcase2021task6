#!/bin/sh

# PYSCRIPT is the name of the python script in $project/standalone, without ".py" extension.
usage="$0 (train|test|download) PARAMS... \nExample: \n\t$0 train path=default bsize=128 epochs=10"

job_name="$1"
fname_script="$1.py"
script_params=`./get_skip_params.sh 1 $@`

path=`./get_param.sh "path" "NOT_FOUND" $@`
if [ "${path}" = "NOT_FOUND" ]; then
  path="olympe"
  script_params="${script_params} path=${path}"
fi

datetime=`./get_param.sh "datetime" "NOT_FOUND" $@`
if [ "${datetime}" = "NOT_FOUND" ]; then
  datetime=`date +"%F_%H:%M:%S"`
  script_params="${script_params} datetime=${datetime}"
fi

tag=`./get_param.sh "tag" "" $@`
cpus=`./get_param.sh "cpus" "4" $@`
gpus=`./get_param.sh "gpus" "1" $@`
dataset=`./get_param.sh "data" "clotho" $@ | tr a-z A-Z`

dpath_project=`realpath $0 | xargs dirname | xargs dirname`
fpath_python="/tmpdir/labbe/miniconda3/envs/env_aac/bin/python"
fpath_script="${dpath_project}/standalone/${fname_script}"

dpath_log="${dpath_project}/logs/sbatch"
fpath_out="${dpath_log}/${dataset}_${job_name}_%j_${tag}.out"
fpath_err="${dpath_log}/${dataset}_${job_name}_%j_${tag}.err"
fpath_singularity=""
srun="srun"

# Build sbatch file ----------------------------------------------------------------------------------------------------
partition=""
# Memory format : number[K|M|G|T]. If 0, no memory limit, use all of node.
mem="64G"
# Time format : days-hours:minutes:seconds. If 0, no time limit.
time="3-00:00:00"

module_load="module load cuda/10.1.105"

fpath_sbatch=".tmp_${job_name}.sbatch"
cat << EOT > ${fpath_sbatch}
#!/bin/sh

# Minimal number of nodes (equiv: -N)
#SBATCH --nodes=1

# Number of tasks (equiv: -n)
#SBATCH --ntasks=1

# Job name (equiv: -J)
#SBATCH --job-name=${job_name}

# Log output file
#SBATCH --output=${fpath_out}

# Log err file
#SBATCH --error=${fpath_err}

# Number of CPU per task
#SBATCH --cpus-per-task=${cpus}

# Memory limit (0 means no limit)
#SBATCH --mem=${mem}

# Duration limit (0 means no limit)
#SBATCH --time=${time}

# Mail for optional auto-sends
#SBATCH --mail-user=etienne.labbe@irit.fr


# For GPU nodes, select partition
#SBATCH --partition=${partition}

# For GPU nodes, select the number of GPUs
#SBATCH --gres=gpu:${gpus}

# For GPU nodes, force job to start only when CPU and GPU are all available
#SBATCH --gres-flags=enforce-binding


# For testing the sbatch file
## #SBATCH --test-only

# Others
## #SBATCH --ntasks-per-node=4
## #SBATCH --ntasks-per-core=1


module purge
${module_load}

${srun} ${fpath_python} ${fpath_script} ${script_params}

EOT

# Run & exit --------------------------------------------------------------------------------------------------------------
echo "Sbatch job '${job_name}' for script '${fname_script}' with tag '${tag}'"
sbatch ${fpath_sbatch}

exit 0
