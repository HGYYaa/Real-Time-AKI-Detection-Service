"""
tests/test_inference.py
Unit tests for the Inference module.
Ensures the Random Forest model correctly predicts AKI based on creatinine history.
"""

import pytest
from src import inference

@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """
    Initializes the model once for the entire test module.
    """
    inference.initialize()

def test_predict_empty_history():
    assert inference.predict('m', []) is False

def test_predict_stable_patient():
    history = [
        ("202401011000", 100.0),
        ("202401021000", 102.0),
        ("202401031000", 101.0)
    ]
    assert inference.predict('f', history) is False

def test_predict_sharp_increase():
    history = [
        ("202401011000", 100.0),
        ("202401031000", 300.0) 
    ]
    assert inference.predict('m', history) is True

def test_predict_gender_robustness():
    """
    Ensures the module handles different gender strings and None values
    using the internal mapping and imputer.
    """
    history = [("202401011000", 100.0), ("202401031000", 300.0)]
    
    assert inference.predict('male', history) is True
    assert inference.predict('FEMALE', history) is True
    assert inference.predict(None, history) is True  # 应该由 imputer 填补
    assert inference.predict('', history) is True    # 应该由 imputer 填补

def test_predict_long_term_fallback():
    history = [
        ("202401011000", 80.0),
        ("202401012200", 130.0) 
    ]
    assert isinstance(inference.predict('f', history), bool)

def test_predict_robustness_to_types():
    history = [
        ("20240101100005", "90.5"), 
        ("20240102100010", 150.0)
    ]
    try:
        result = inference.predict('m', history)
        assert isinstance(result, bool)
    except Exception as e:
        pytest.fail(f"predict() crashed on unconventional input formats: {e}")