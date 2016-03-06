from datacleaner import autoclean, autoclean_cv
import pandas as pd
import numpy as np

np.random.seed(300)

def test_autoclean_already_clean_data():
    """Test autoclean() with already-clean data"""
    data = pd.DataFrame({'A': np.random.rand(1000),
                         'B': np.random.rand(1000),
                         'C': np.random.randint(0, 3, 1000)})

    cleaned_data = autoclean(data)

    # autoclean() should not change the data at all
    assert cleaned_data.equals(data)

def test_autoclean_cv_already_clean_data():
    """Test autoclean_cv() with already-clean data"""
    data = pd.DataFrame({'A': np.random.rand(1000),
                         'B': np.random.rand(1000),
                         'C': np.random.randint(0, 3, 1000)})

    training_data = data[:500].copy()
    testing_data = data[500:].copy()

    cleaned_training_data, cleaned_testing_data = autoclean_cv(training_data, testing_data)

    # autoclean_cv() should not change the data at all
    assert cleaned_training_data.equals(training_data)
    assert cleaned_testing_data.equals(testing_data)

def test_autoclean_with_nans_all_numerical():
    data = pd.DataFrame({'A': np.random.rand(1000),
                         'B': np.random.rand(1000),
                         'C': np.random.randint(0, 3, 1000)})

    data.loc[10:20, 'A'] = np.nan
    data.loc[50:70, 'C'] = np.nan

    hand_cleaned_data = data.copy()
    hand_cleaned_data['A'].fillna(hand_cleaned_data['A'].median(), inplace=True)
    hand_cleaned_data['C'].fillna(hand_cleaned_data['C'].mode()[0], inplace=True)

    cleaned_data = autoclean(data)

    assert cleaned_data.equals(hand_cleaned_data)

def test_autoclean_cv_with_nans_all_numerical():
    data = pd.DataFrame({'A': np.random.rand(1000),
                         'B': np.random.rand(1000),
                         'C': np.random.randint(0, 3, 1000)})
    
    training_data = data[:500].copy()
    testing_data = data[500:].copy()

    training_data.loc[10:20, 'A'] = np.nan
    training_data.loc[50:70, 'C'] = np.nan

    testing_data.loc[70:80, 'A'] = np.nan
    testing_data.loc[10:40, 'C'] = np.nan

    hand_cleaned_training_data = training_data.copy()
    hand_cleaned_testing_data = testing_data.copy()

    hand_cleaned_training_data['A'].fillna(hand_cleaned_training_data['A'].median(), inplace=True)
    hand_cleaned_training_data['C'].fillna(hand_cleaned_training_data['C'].mode()[0], inplace=True)

    hand_cleaned_testing_data['A'].fillna(hand_cleaned_testing_data['A'].median(), inplace=True)
    hand_cleaned_testing_data['C'].fillna(hand_cleaned_testing_data['C'].mode()[0], inplace=True)

    cleaned_training_data, cleaned_testing_data = autoclean_cv(training_data, testing_data)

    assert cleaned_training_data.equals(hand_cleaned_training_data)
    assert cleaned_testing_data.equals(hand_cleaned_testing_data)
