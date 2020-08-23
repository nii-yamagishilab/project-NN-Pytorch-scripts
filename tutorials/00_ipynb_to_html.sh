#!/usr/bin/sh

for name in `ls *.ipynb`
do
    echo ${name}
    jupyter nbconvert --to html ${name}
done
