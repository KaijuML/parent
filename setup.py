# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
# Copyright 2021 Clément Rebuffel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import find_packages, setup
import pkg_resources
import os


PACKAGE_NAME = 'parent'
VERSION = '1.1.1'

KEYWORDS = ' '.join([
    'parent', 'metric', 'evaluation', 'data-to-text', 'nlg', 'ngram'
])


# For more info on CLASSIFIERS: https://pypi.org/classifiers/
# For a readable list: https://pypi.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Science/Research'
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Text Processing :: Linguistic',
]

this_directory = pkg_resources.resource_filename(__name__, '.')
with open(os.path.join(this_directory, 'README.md'), encoding='utf8') as f:
    long_description = f.read()


setup(
    author='Clément Rebuffel',
    author_email='clement.rebuffel@lip6.fr',
    maintainer='Clément Rebuffel',
    maintainer_email='clement.rebuffel@lip6.fr',
    
    name=PACKAGE_NAME,
    version=VERSION,
    
    description='Compute PARENT metric (from Dhingra et al. 2019)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    license='Apache',
    keywords=KEYWORDS,
    url='https://github.com/KaijuML/parent',
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    
    python_requires=">=3",
    install_requires=[
        'numpy',
        'tqdm'
    ],
    
    scripts=['parent/parent.py'],
    entry_points={
        "console_scripts": [
            "parent=parent:main",
        ]
    }
)
