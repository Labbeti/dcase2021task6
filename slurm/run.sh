#!/bin/sh

# PYSCRIPT is the name of the python script in $project/standalone, without ".py" extension.
usage="$0 (train|test|download) [path=(default|osirim|olympe)] PARAMS... \nExample: \n\t$0 train path=default bsize=128 epochs=10"

dpath_sbatch=`realpath $0 | xargs dirname`
path=`${dpath_sbatch}/get_param.sh "path" "" $@`

if [ "${path}" = "default" ]; then
	${dpath_sbatch}/run_default.sh $@
elif [ "${path}" = "osirim" ]; then
	${dpath_sbatch}/run_osirim.sh $@
elif [ "${path}" = "olympe" ]; then
	${dpath_sbatch}/run_olympe.sh $@
else
	echo "${usage}"
fi

exit 0
