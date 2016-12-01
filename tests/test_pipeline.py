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


def test_mergerResolvingTestDatasetNewLabelImage():
    check_call(["python",
                "../hytra/configtemplates/create_config.py",
                "--in",
                "../tests/mergerResolvingTestDatasetNewLabelImage/config_template.ini",
                "--out",
                "../tests/mergerResolvingTestDatasetNewLabelImage/test_config.ini",
                "embryonicDir",
                ".."
                ])
    check_call(["python", "pipeline.py", "--config", "../tests/mergerResolvingTestDatasetNewLabelImage/test_config.ini"])

    for f in range(4):
        frame = vigra.impex.readImage('../tests/mergerResolvingTestDatasetNewLabelImage/ctc_RES/mask00{}.tif'.format(f))
        assert(np.all(np.unique(frame) == [0,1,2]))

    with open('../tests/mergerResolvingTestDatasetNewLabelImage/ctc_RES/res_track.txt', 'r') as resultFile:
        assert(resultFile.read() == '1 0 3 0\n2 0 3 0\n')

def test_mergerResolvingTestDataset_withoutTracklets():
    check_call(["python",
                "../hytra/configtemplates/create_config.py",
                "--in",
                "../tests/mergerResolvingTestDataset/config_template.ini",
                "--out",
                "../tests/mergerResolvingTestDataset/test_config.ini",
                "embryonicDir",
                ".."
                ])
    check_call(["python", "pipeline.py", "--config", "../tests/mergerResolvingTestDataset/test_config.ini", "--without-tracklets"])

    for f in range(4):
        frame = vigra.impex.readImage('../tests/mergerResolvingTestDataset/ctc_RES/mask00{}.tif'.format(f))
        assert(np.all(np.unique(frame) == [0,1,2]))

    with open('../tests/mergerResolvingTestDataset/ctc_RES/res_track.txt', 'r') as resultFile:
        assert(resultFile.read() == '1 0 3 0\n2 0 3 0\n')

def test_divisionTestDataset():
    check_call(["python",
                "../hytra/configtemplates/create_config.py",
                "--in",
                "../tests/divisionTestDataset/config_template.ini",
                "--out",
                "../tests/divisionTestDataset/test_config.ini",
                "embryonicDir",
                ".."
                ])
    check_call(["python", "pipeline.py", "--config", "../tests/divisionTestDataset/test_config.ini"])

    expectedIds = [[0, 1], [0, 1], [0, 2, 3], [0, 2, 3]]
    for f in range(4):
        frame = vigra.impex.readImage('../tests/divisionTestDataset/ctc_RES/mask00{}.tif'.format(f))
        assert(np.all(np.unique(frame) == expectedIds[f]))

    with open('../tests/divisionTestDataset/ctc_RES/res_track.txt', 'r') as resultFile:
        assert(resultFile.read() == '1 0 1 0\n2 2 3 1\n3 2 3 1\n')

def test_divisionTestDataset_withoutTracklets():
    check_call(["python",
                "../hytra/configtemplates/create_config.py",
                "--in",
                "../tests/divisionTestDataset/config_template.ini",
                "--out",
                "../tests/divisionTestDataset/test_config.ini",
                "embryonicDir",
                ".."
                ])
    check_call(["python", "pipeline.py", "--config", "../tests/divisionTestDataset/test_config.ini", "--without-tracklets"])

    expectedIds = [[0, 1], [0, 1], [0, 2, 3], [0, 2, 3]]
    for f in range(4):
        frame = vigra.impex.readImage('../tests/divisionTestDataset/ctc_RES/mask00{}.tif'.format(f))
        assert(np.all(np.unique(frame) == expectedIds[f]))

    with open('../tests/divisionTestDataset/ctc_RES/res_track.txt', 'r') as resultFile:
        assert(resultFile.read() == '1 0 1 0\n2 2 3 1\n3 2 3 1\n')

def test_noscripts_mergerResolvingTestDataset():
    check_call(["python",
                "../hytra/configtemplates/create_config.py",
                "--in",
                "../tests/mergerResolvingTestDataset/config_template.ini",
                "--out",
                "../tests/mergerResolvingTestDataset/test_config.ini",
                "embryonicDir",
                ".."
                ])
    check_call(["python", "pipelineNoScripts.py", "--config", "../tests/mergerResolvingTestDataset/test_config.ini"])

    for f in range(4):
        frame = vigra.impex.readImage('../tests/mergerResolvingTestDataset/ctc_RES/mask00{}.tif'.format(f))
        assert(np.all(np.unique(frame) == [0,1,2]))

    with open('../tests/mergerResolvingTestDataset/ctc_RES/res_track.txt', 'r') as resultFile:
        assert(resultFile.read() == '1 0 3 0\n2 0 3 0\n')

def test_boxesTestDataset():
    check_call(["python",
                "../hytra/configtemplates/create_config.py",
                "--in",
                "../tests/boxesTestDataset/config_template.ini",
                "--out",
                "../tests/boxesTestDataset/test_config.ini",
                "embryonicDir",
                ".."
                ])
    check_call(["python", "pipeline.py", "--config", "../tests/boxesTestDataset/test_config.ini"])

    expectedIds = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3, 4, 5], [0, 2, 3, 4, 5]]
    for f in range(4):
        frame = vigra.impex.readImage('../tests/boxesTestDataset/ctc_RES/mask00{}.tif'.format(f))
        assert(np.all(np.unique(frame) == expectedIds[f]))

    with open('../tests/boxesTestDataset/ctc_RES/res_track.txt', 'r') as resultFile:
        assert(resultFile.read() == '1 0 4 0\n2 0 6 0\n3 0 6 0\n4 5 6 1\n5 5 6 1\n')

if __name__ == "__main__":
    test_divisionTestDataset()
    test_mergerResolvingTestDataset()
    test_divisionTestDataset_withoutTracklets()
    test_mergerResolvingTestDataset_withoutTracklets()
    test_noscripts_mergerResolvingTestDataset()
    test_boxesTestDataset()
