[![Build Status](https://travis-ci.org/rhiever/datacleaner.svg?branch=master)](https://travis-ci.org/rhiever/datacleaner)
[![Code Health](https://landscape.io/github/rhiever/datacleaner/master/landscape.svg?style=flat)](https://landscape.io/github/rhiever/datacleaner/master)
[![Coverage Status](https://coveralls.io/repos/github/rhiever/datacleaner/badge.svg?branch=master)](https://coveralls.io/github/rhiever/datacleaner?branch=master)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![License](https://img.shields.io/badge/license-MIT%20License-blue.svg)
[![PyPI version](https://badge.fury.io/py/datacleaner.svg)](https://badge.fury.io/py/datacleaner)


# datacleaner

[![Join the chat at https://gitter.im/rhiever/datacleaner](https://badges.gitter.im/rhiever/datacleaner.svg)](https://gitter.im/rhiever/datacleaner?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

A Python tool that automatically cleans data sets and readies them for analysis.

## datacleaner is not magic

datacleaner works with data in [pandas DataFrames](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).

datacleaner is not magic, and it won't take an unorganized blob of text and automagically parse it out for you.

What datacleaner *will* do is save you a ton of time encoding and cleaning your data once it's already in a format that pandas DataFrames can handle.

Currently, datacleaner does the following:

* Optionally drops any row with a missing value

* Replaces missing values with the mode (for categorical variables) or median (for continuous variables) on a column-by-column basis

* Encodes non-numerical variables (e.g., categorical variables with strings) with numerical equivalents

We plan to add more cleaning features as the project grows.

## License

Please see the [repository license](https://github.com/rhiever/datacleaner/blob/master/LICENSE) for the licensing and usage information for datacleaner.

Generally, we have licensed datacleaner to make it as widely usable as possible.

## Installation

datacleaner is built to use pandas DataFrames and some scikit-learn modules for data preprocessing. As such, we recommend installing the [Anaconda Python distribution](https://www.continuum.io/downloads) prior to installing datacleaner.

Once the prerequisites are installed, datacleaner can be installed with a simple `pip` command:

```
pip install datacleaner
```

## Usage

### datacleaner on the command line

datacleaner can be used on the command line. Use `--help` to see its usage instructions.

```
usage: datacleaner [-h] [-cv CROSS_VAL_FILENAME] [-o OUTPUT_FILENAME]
                   [-cvo CV_OUTPUT_FILENAME] [-is INPUT_SEPARATOR]
                   [-os OUTPUT_SEPARATOR] [--drop-nans]
                   [--ignore-update-check] [--version]
                   INPUT_FILENAME

A Python tool that automatically cleans data sets and readies them for analysis

positional arguments:
  INPUT_FILENAME        File name of the data file to clean

optional arguments:
  -h, --help            show this help message and exit
  -cv CROSS_VAL_FILENAME
                        File name for the validation data set if performing
                        cross-validation
  -o OUTPUT_FILENAME    Data file to output the cleaned data set to
  -cvo CV_OUTPUT_FILENAME
                        Data file to output the cleaned cross-validation data
                        set to
  -is INPUT_SEPARATOR   Column separator for the input file(s) (default: \t)
  -os OUTPUT_SEPARATOR  Column separator for the output file(s) (default: \t)
  --drop-nans           Drop all rows that have a NaN in any column (default: False)
  --ignore-update-check
                        Do not check for the latest version of datacleaner
                        (default: False)
  --version             show program's version number and exit
```

An example command-line call to datacleaner may look like:

```
datacleaner my_data.csv -o my_clean.data.csv -is , -os ,
```

which will read the data from `my_data.csv` (assuming columns are separated by commas), clean the data set, then output the resulting data set to `my_clean.data.csv`.

### datacleaner in scripts

datacleaner can also be used as part of a script. There are two primary functions implemented in datacleaner: `autoclean` and `autoclean_cv`.

```
autoclean(input_dataframe, drop_nans=False, copy=False, ignore_update_check=False)
    Performs a series of automated data cleaning transformations on the provided data set
    
    Parameters
    ----------
    input_dataframe: pandas.DataFrame
        Data set to clean
    drop_nans: bool
        Drop all rows that have a NaN in any column (default: False)
    copy: bool
        Make a copy of the data set (default: False) 
    encoder: category_encoders transformer
        The a valid category_encoders transformer which is passed an inferred cols list. Default (None: LabelEncoder)
    encoder_kwargs: category_encoders
        The a valid sklearn transformer to encode categorical features. Default (None)
    ignore_update_check: bool
        Do not check for the latest version of datacleaner

    Returns
    ----------
    output_dataframe: pandas.DataFrame
        Cleaned data set
```

```
autoclean_cv(training_dataframe, testing_dataframe, drop_nans=False, copy=False, ignore_update_check=False)
    Performs a series of automated data cleaning transformations on the provided training and testing data sets
    
    Unlike `autoclean()`, this function takes cross-validation into account by learning the data transformations
    from only the training set, then applying those transformations to both the training and testing set.
    By doing so, this function will prevent information leak from the training set into the testing set.
    
    Parameters
    ----------
    training_dataframe: pandas.DataFrame
        Training data set
    testing_dataframe: pandas.DataFrame
        Testing data set
    drop_nans: bool
        Drop all rows that have a NaN in any column (default: False)
    copy: bool
        Make a copy of the data set (default: False)  
    encoder: category_encoders transformer
        The a valid category_encoders transformer which is passed an inferred cols list. Default (None: LabelEncoder)
    encoder_kwargs: category_encoders
        The a valid sklearn transformer to encode categorical features. Default (None)
    ignore_update_check: bool
        Do not check for the latest version of datacleaner

    Returns
    ----------
    output_training_dataframe: pandas.DataFrame
        Cleaned training data set
    output_testing_dataframe: pandas.DataFrame
        Cleaned testing data set
```

Below is an example of datacleaner performing basic cleaning on a data set.

```python
from datacleaner import autoclean
import pandas as pd

my_data = pd.read_csv('my_data.csv', sep=',')
my_clean_data = autoclean(my_data)
my_data.to_csv('my_clean_data.csv', sep=',', index=False)
```

Note that because datacleaner works directly on [pandas DataFrames](http://pandas.pydata.org/pandas-docs/stable/10min.html), all [DataFrame operations](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) are still available to the resulting data sets.

## Contributing to datacleaner

We welcome you to [check the existing issues](https://github.com/rhiever/datacleaner/issues/) for bugs or enhancements to work on. If you have an idea for an extension to datacleaner, please [file a new issue](https://github.com/rhiever/datacleaner/issues/new) so we can discuss it.

## Citing datacleaner

If you use datacleaner as part of your workflow in a scientific publication, please consider citing the datacleaner repository with the following DOI:

[![DOI](https://zenodo.org/badge/20747/rhiever/datacleaner.svg)](https://zenodo.org/badge/latestdoi/20747/rhiever/datacleaner)
