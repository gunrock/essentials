#!/bin/bash
exe_file="/home/mdresche/essentials/build/bin/spmv"
DATADIR="/data/gunrock/gunrock_dataset/luigi-8TB/large/"

test_datasets=()
ALL_IN_DIR="${DATADIR}*"
for f in $ALL_IN_DIR
do 
    if [ -d "$f" ]; then 
        dir_name=$(basename "${f}")
        test_datasets=(${test_datasets[*]} "$dir_name")
    fi
done
i=0
for dataset in  "${test_datasets[@]}"
do
    $exe_file "reorder" $DATADIR$dataset/$dataset.mtx
    $exe_file "random" $DATADIR$dataset/$dataset.mtx
    ((i++))
    if [[ $i -gt 2 ]]
    then
	break
    else
	echo "hi"
    fi
      
done
