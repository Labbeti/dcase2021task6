#!/bin/sh

# PYSCRIPT is the name of the python script in $project/standalone, without ".py" extension.
usage="$0 (train|test|download) PARAMS... \nExample: \n\t$0 train path=default bsize=128 epochs=10"

cwd=`realpath $0 | xargs dirname`
fname_script="$1.py"
script_params=`${cwd}/get_skip_params.sh 1 $@`

path=`${cwd}/get_param.sh "path" "NOT_FOUND" $@`
if [ "${path}" = "NOT_FOUND" ]; then
  path="default"
  script_params="${script_params} path=${path}"
fi

datetime=`${cwd}/get_param.sh "datetime" "NOT_FOUND" $@`
if [ "${datetime}" = "NOT_FOUND" ]; then
  datetime=`date +"%F_%H:%M:%S"`
  script_params="${script_params} datetime=${datetime}"
fi

fpath_python="python"
fpath_script="${cwd}/../standalone/${fname_script}"

tag=`${cwd}/get_param.sh "tag" "" $@`
echo "Run script '${fname_script}' with tag '${tag}'"
${fpath_python} ${fpath_script} ${script_params}

exit 0
