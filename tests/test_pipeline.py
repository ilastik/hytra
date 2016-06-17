# pythonpath modification to make hytra available 
# for import without requiring it to be installed
import os
os.chdir('scripts')
import sys
sys.path.insert(0, os.path.abspath('..'))
from subprocess import check_call
import vigra
import numpy as np

def test_mergerResolvingTestDataset():
    check_call(["python",
                "../hytra/configtemplates/create_config.py",
                "--in",
                "../tests/mergerResolvingTestDataset/config_template.ini",
                "--out",
                "../tests/mergerResolvingTestDataset/test_config.ini",
                "embryonicDir",
                ".."
                ])
    check_call(["python", "pipeline.py", "--config", "../tests/mergerResolvingTestDataset/test_config.ini"])

    for f in range(4):
        frame = vigra.impex.readImage('../tests/mergerResolvingTestDataset/ctc_RES/mask00{}.tif'.format(f))
        assert(np.all(np.unique(frame) == [0,1,2]))

    with open('../tests/mergerResolvingTestDataset/ctc_RES/res_track.txt', 'r') as resultFile:
        assert(resultFile.read() == '1 0 3 0\n2 0 3 0\n')