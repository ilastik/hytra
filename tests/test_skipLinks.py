# pythonpath modification to make hytra available 
# for import without requiring it to be installed
import os

SCRIPTS_PATH = 'scripts'
assert os.path.exists(SCRIPTS_PATH), 'Scripts path not found (Maybe you forgot to run the test from the root directory?)'
os.chdir(SCRIPTS_PATH)

import sys
sys.path.insert(0, os.path.abspath('..'))
from subprocess import check_call
import vigra
import numpy as np

def test_skipLinksTestDataset_withoutTracklets():
    check_call(["python",
                "../hytra/configtemplates/create_config.py",
                "--in",
                "../tests/skipLinksTestDataset/config_template.ini",
                "--out",
                "../tests/skipLinksTestDataset/test_config.ini",
                "embryonicDir",
                "..",
                ])
    check_call(["python", "pipeline_skip_links.py", "--config", "../tests/skipLinksTestDataset/test_config.ini"])

    for f in range(4):
        frame = vigra.impex.readImage('../tests/skipLinksTestDataset/ctc_RES/mask00{}.tif'.format(f))
        if f == 1: # the second frame is empty
            assert(np.all(np.unique(frame) == [0]))
        elif f==0:
            assert(np.all(np.unique(frame) == [0, 1]))
        else:
            assert(np.all(np.unique(frame) == [0, 2]))


    with open('../tests/skipLinksTestDataset/ctc_RES/res_track.txt', 'r') as resultFile:
        assert(resultFile.read() == '1 0 0 0\n2 2 3 1\n')

if __name__ == "__main__":
    test_skipLinksTestDataset_withoutTracklets()