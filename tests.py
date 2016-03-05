from datacleaner import autoclean, autoclean_cv
import pandas as pd
import numpy as np

def test_autoclean_already_clean_data():
    """Test autoclean() with already-clean data"""
    data = pd.DataFrame({'A': np.random.rand(1000),
                         'B': np.random.rand(1000),
                         'C': np.random.randint(0, 3, 1000)})

    clean_data = autoclean(data)

    # autoclean() should not change the data at all
    assert clean_data.equals(data)

def test_autoclean_cv_already_clean_data():
    """Test autoclean_cv() with already-clean data"""
    data = pd.DataFrame({'A': np.random.rand(1000),
                         'B': np.random.rand(1000),
                         'C': np.random.randint(0, 3, 1000)})

    training_data = data[:500].copy()
    testing_data = data[500:].copy()

    clean_training_data, clean_testing_data = autoclean_cv(training_data, testing_data)

    # autoclean_cv() should not change the data at all
    assert clean_training_data.equals(training_data)
    assert clean_testing_data.equals(testing_data)
