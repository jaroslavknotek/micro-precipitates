#!/usr/bin/env python

from distutils.core import setup

setup(name='MicroPrec',
    version='0.2',
    description='Prec -> distribution',
    author='Jaroslav Knotek',
    author_email='knotekjaroslav@email.cz',
    url='https://github.com/jaroslavknotek/micro-precipitates',
    packages=['precipitates'],
#     python_requires='>=3.9',
    install_requires=[
        'numpy',
        'opencv-python',
        'tqdm',
        'imageio',
        'scikit-image',
        'scikit-learn',
        'tensorflow',
        'matplotlib',
        'pandas',
        'streamlit'
    ]
)
