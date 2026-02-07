#!/usr/bin/bash

input_file_list=
datadir=
savedir=

# to parallize the jobs
split -l 8000 ${input_file_list}

for filelist in `ls xa*`
do
    qsub -g ${gid} -o log_${filelist} -e log_${filelist}_err submit.sh ${filelist} ${datadir} ${savedir}
done

