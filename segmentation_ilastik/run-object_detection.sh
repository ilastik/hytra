#! /usr/bin/env bash

PREFIX="/home/mschiegg/software/ilastik.rc28917"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PREFIX/lib
export PATH=$PATH:$PREFIX/bin
export PYTHONPATH=$PYTHONPATH:$PREFIX:$PREFIX/lib:$PREFX/lib/python2.7/site-packages

python /home/mschiegg/embryonic/segmentation_ilastik/object_detection.py $@
