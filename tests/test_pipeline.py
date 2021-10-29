# pythonpath modification to make hytra available
# for import without requiring it to be installed
import os

SCRIPTS_PATH = "scripts"
assert os.path.exists(
    SCRIPTS_PATH
), "Scripts path not found (Maybe you forgot to run the test from the root directory?)"
os.chdir(SCRIPTS_PATH)

import sys
from pathlib import Path

from subprocess import check_call
import vigra
import numpy as np


test_file_path = Path(__file__).parent


def test_mergerResolvingTestDataset():
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    test_file_path.parent / "hytra" / "configtemplates" / "create_config.py",
                    "--in",
                    test_file_path / "mergerResolvingTestDataset" / "config_template.ini",
                    "--out",
                    test_file_path / "mergerResolvingTestDataset" / "test_config.ini",
                    "embryonicDir",
                    "..",
                ],
            )
        )
    )
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    "pipeline.py",
                    "--config",
                    test_file_path / "mergerResolvingTestDataset" / "test_config.ini",
                ],
            )
        )
    )

    for f in range(4):
        frame = vigra.impex.readImage(str(test_file_path / "mergerResolvingTestDataset" / "ctc_RES" / f"mask00{f}.tif"))
        assert np.all(np.unique(frame) == [0, 1, 2])

    resultFile = test_file_path / "mergerResolvingTestDataset" / "ctc_RES" / "res_track.txt"
    assert resultFile.read_text() == "1 0 3 0\n2 0 3 0\n"


def test_mergerResolvingTestDatasetNewLabelImage():
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    test_file_path.parent / "hytra" / "configtemplates" / "create_config.py",
                    "--in",
                    test_file_path / "mergerResolvingTestDatasetNewLabelImage" / "config_template.ini",
                    "--out",
                    test_file_path / "mergerResolvingTestDatasetNewLabelImage" / "test_config.ini",
                    "embryonicDir",
                    "..",
                ],
            )
        )
    )
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    "pipeline.py",
                    "--config",
                    test_file_path / "mergerResolvingTestDatasetNewLabelImage" / "test_config.ini",
                ],
            )
        )
    )

    for f in range(4):
        frame = vigra.impex.readImage(
            str(test_file_path / "mergerResolvingTestDatasetNewLabelImage" / "ctc_RES" / f"mask00{f}.tif")
        )
        assert np.all(np.unique(frame) == [0, 1, 2])

    resultFile = test_file_path / "mergerResolvingTestDatasetNewLabelImage" / "ctc_RES" / "res_track.txt"
    assert resultFile.read_text() == "1 0 3 0\n2 0 3 0\n"


def test_mergerResolvingTestDataset_withoutTracklets():
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    test_file_path.parent / "hytra" / "configtemplates" / "create_config.py",
                    "--in",
                    test_file_path / "mergerResolvingTestDataset" / "config_template.ini",
                    "--out",
                    test_file_path / "mergerResolvingTestDataset" / "test_config.ini",
                    "embryonicDir",
                    "..",
                ],
            )
        )
    )
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    "pipeline.py",
                    "--config",
                    str(test_file_path / "mergerResolvingTestDataset" / "test_config.ini"),
                    "--without-tracklets",
                ],
            )
        )
    )

    for f in range(4):
        frame = vigra.impex.readImage(str(test_file_path / "mergerResolvingTestDataset" / "ctc_RES" / f"mask00{f}.tif"))
        assert np.all(np.unique(frame) == [0, 1, 2])

    resultFile = test_file_path / "mergerResolvingTestDataset" / "ctc_RES" / "res_track.txt"
    assert resultFile.read_text() == "1 0 3 0\n2 0 3 0\n"


def test_divisionTestDataset():
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    test_file_path.parent / "hytra" / "configtemplates" / "create_config.py",
                    "--in",
                    test_file_path / "divisionTestDataset" / "config_template.ini",
                    "--out",
                    test_file_path / "divisionTestDataset" / "test_config.ini",
                    "embryonicDir",
                    "..",
                ],
            )
        )
    )
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    "pipeline.py",
                    "--config",
                    str(test_file_path / "divisionTestDataset" / "test_config.ini"),
                ],
            )
        )
    )

    expectedIds = [[0, 1], [0, 1], [0, 2, 3], [0, 2, 3]]
    for f in range(4):
        frame = vigra.impex.readImage(str(test_file_path / "divisionTestDataset" / "ctc_RES" / f"mask00{f}.tif"))
        assert np.all(np.unique(frame) == expectedIds[f])

    resultFile = test_file_path / "divisionTestDataset" / "ctc_RES" / "res_track.txt"
    assert resultFile.read_text() == "1 0 1 0\n2 2 3 1\n3 2 3 1\n"


def test_divisionTestDataset_withoutTracklets():
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    test_file_path.parent / "hytra" / "configtemplates" / "create_config.py",
                    "--in",
                    test_file_path / "divisionTestDataset" / "config_template.ini",
                    "--out",
                    test_file_path / "divisionTestDataset" / "test_config.ini",
                    "embryonicDir",
                    "..",
                ],
            )
        )
    )
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    "pipeline.py",
                    "--config",
                    str(test_file_path / "divisionTestDataset" / "test_config.ini"),
                    "--without-tracklets",
                ],
            )
        )
    )

    expectedIds = [[0, 1], [0, 1], [0, 2, 3], [0, 2, 3]]
    for f in range(4):
        frame = vigra.impex.readImage(str(test_file_path / "divisionTestDataset" / "ctc_RES" / f"mask00{f}.tif"))
        assert np.all(np.unique(frame) == expectedIds[f])

    resultFile = test_file_path / "divisionTestDataset" / "ctc_RES" / "res_track.txt"
    assert resultFile.read_text() == "1 0 1 0\n2 2 3 1\n3 2 3 1\n"


def test_noscripts_mergerResolvingTestDataset():
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    test_file_path.parent / "hytra" / "configtemplates" / "create_config.py",
                    "--in",
                    test_file_path / "mergerResolvingTestDataset" / "config_template.ini",
                    "--out",
                    test_file_path / "mergerResolvingTestDataset" / "test_config.ini",
                    "embryonicDir",
                    "..",
                ],
            )
        )
    )
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    "pipelineNoScripts.py",
                    "--config",
                    str(test_file_path / "mergerResolvingTestDataset" / "test_config.ini"),
                ],
            )
        )
    )

    for f in range(4):
        frame = vigra.impex.readImage(str(test_file_path / "mergerResolvingTestDataset" / "ctc_RES" / f"mask00{f}.tif"))
        assert np.all(np.unique(frame) == [0, 1, 2])

    resultFile = test_file_path / "mergerResolvingTestDataset" / "ctc_RES" / "res_track.txt"
    assert resultFile.read_text() == "1 0 3 0\n2 0 3 0\n"


def test_skipLinksTestDataset_withoutTracklets():
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    test_file_path.parent / "hytra" / "configtemplates" / "create_config.py",
                    "--in",
                    test_file_path / "skipLinksTestDataset" / "config_template.ini",
                    "--out",
                    test_file_path / "skipLinksTestDataset" / "test_config.ini",
                    "embryonicDir",
                    "..",
                ],
            )
        )
    )
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    "pipeline_skip_links.py",
                    "--config",
                    str(test_file_path / "skipLinksTestDataset" / "test_config.ini"),
                ],
            )
        )
    )

    for f in range(4):
        frame = vigra.impex.readImage(str(test_file_path / "skipLinksTestDataset" / "ctc_RES" / f"mask00{f}.tif"))
        if f == 1:  # the second frame is empty
            assert np.all(np.unique(frame) == [0])
        elif f == 0:
            assert np.all(np.unique(frame) == [0, 1])
        else:
            assert np.all(np.unique(frame) == [0, 2])

        resultFile = test_file_path / "skipLinksTestDataset" / "ctc_RES" / "res_track.txt"
        assert resultFile.read_text() == "1 0 0 0\n2 2 3 1\n"


def test_boxesTestDataset():
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    test_file_path.parent / "hytra" / "configtemplates" / "create_config.py",
                    "--in",
                    test_file_path / "boxesTestDataset" / "config_template.ini",
                    "--out",
                    test_file_path / "boxesTestDataset" / "test_config.ini",
                    "embryonicDir",
                    "..",
                ],
            )
        )
    )
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    "pipeline.py",
                    "--config",
                    str(test_file_path / "boxesTestDataset" / "test_config.ini"),
                ],
            )
        )
    )

    expectedIds = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 2, 3, 4, 5],
        [0, 2, 3, 4, 5],
    ]
    for f in range(4):
        frame = vigra.impex.readImage(str(test_file_path / "boxesTestDataset" / "ctc_RES" / f"mask00{f}.tif"))
        assert np.all(np.unique(frame) == expectedIds[f])

    resultFile = test_file_path / "boxesTestDataset" / "ctc_RES" / "res_track.txt"
    assert resultFile.read_text() == "1 0 4 0\n2 0 6 0\n3 0 6 0\n4 5 6 1\n5 5 6 1\n"


def test_divisionMergerTestDataset():
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    test_file_path.parent / "hytra" / "configtemplates" / "create_config.py",
                    "--in",
                    test_file_path / "divisionMergerTestDataset" / "config_template.ini",
                    "--out",
                    test_file_path / "divisionMergerTestDataset" / "test_config.ini",
                    "embryonicDir",
                    "..",
                ],
            )
        )
    )
    check_call(
        list(
            map(
                str,
                [
                    "python",
                    "pipeline.py",
                    "--config",
                    str(test_file_path / "divisionMergerTestDataset" / "test_config.ini"),
                ],
            )
        )
    )

    expectedIds = [[0, 1, 2, 3], [0, 1, 3, 4, 5], [0, 1, 3, 4, 5]]
    for f in range(3):
        frame = vigra.impex.readImage(str(test_file_path / "divisionMergerTestDataset" / "ctc_RES" / f"mask00{f}.tif"))
        assert np.all(np.unique(frame) == expectedIds[f])

    resultFile = test_file_path / "divisionMergerTestDataset" / "ctc_RES" / "res_track.txt"
    assert resultFile.read_text() == "1 0 2 0\n2 0 0 0\n3 0 2 0\n4 1 2 2\n5 1 2 2\n"


if __name__ == "__main__":
    test_divisionTestDataset()
    test_mergerResolvingTestDataset()
    test_divisionTestDataset_withoutTracklets()
    test_mergerResolvingTestDataset_withoutTracklets()
    test_noscripts_mergerResolvingTestDataset()
    test_skipLinksTestDataset_withoutTracklets()
    test_boxesTestDataset()
    test_divisionMergerTestDataset()
