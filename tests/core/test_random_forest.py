from hytra.core.probabilitygenerator import RandomForestClassifier


def test_rf():
    rf = RandomForestClassifier("/CountClassification", "tests/mergerResolvingTestDataset/tracking.ilp")
    assert len(rf._randomForests) == 1
    assert len(rf.selectedFeatures) == 4
