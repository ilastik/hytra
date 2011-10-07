#!/bin/bash

for i in $( find `cd $1; pwd` | sort ); do
    if [ -f $i ]; then
	prefix=$(basename $i .tif)
	mkdir $prefix
	cd $prefix
	tiffsplit $i $prefix
	cd ..
	echo $i
     fi
done