from hytra.core.probabilitygenerator import RandomForestClassifier
from pathlib import Path

tests_root = Path(__file__).parent.parent

def test_rf():
    rf = RandomForestClassifier("/CountClassification", tests_root / "mergerResolvingTestDataset" / "tracking.ilp")
    assert len(rf._randomForests) == 1
    assert len(rf.selectedFeatures) == 4
