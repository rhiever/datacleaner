#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

def calculate_version():
    initpy = open('datacleaner/_version.py').read().split('\n')
    version = list(filter(lambda x: '__version__' in x, initpy))[0].split('\'')[1]
    return version

package_version = calculate_version()

setup(
    name='datacleaner',
    version=package_version,
    author='Randal S. Olson',
    author_email='rso@randalolson.com',
    packages=find_packages(),
    url='https://github.com/rhiever/datacleaner',
    license='License :: OSI Approved :: MIT License',
    entry_points={'console_scripts': ['datacleaner=datacleaner:main', ]},
    description=('A Python tool that automatically cleans data sets and readies them for analysis.'),
    long_description='''
A Python tool that automatically cleans data sets and readies them for analysis.

Contact
=============
If you have any questions or comments about datacleaner, please feel free to contact me via:

E-mail: rso@randalolson.com

or Twitter: https://twitter.com/randal_olson

This project is hosted at https://github.com/rhiever/datacleaner
''',
    zip_safe=True,
    install_requires=['pandas', 'scikit-learn', 'update_checker'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Utilities'
    ],
    keywords=['data cleaning', 'csv', 'machine learning', 'data analysis', 'data engineering'],
)
