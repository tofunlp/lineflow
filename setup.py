#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distuils.core import setup


setup(
    name='lineflow',
    version='0.1.2',
    description='Framework-Agnostic NLP Data Pipeline in Python',
    long_description=open('./README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yasufumy/lineflow',
    author='Yasufumi Taniguchi',
    author_email='yasufumi.taniguchi@gmail.com',
    packages=[
        'lineflow'
    ],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    tests_require=['pytest']
)
