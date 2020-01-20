#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='lineflow',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='Lightweight NLP Data Loader for All Deep Learning Frameworks in Python',
    long_description=open('./README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yasufumy/lineflow',
    author='Yasufumi Taniguchi',
    author_email='yasufumi.taniguchi@gmail.com',
    packages=[
        'lineflow', 'lineflow.datasets'
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=['arrayfiles', 'gdown'],
    tests_require=['pytest'],
)
