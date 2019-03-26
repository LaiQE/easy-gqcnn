# -*- coding: utf-8 -*-
"""
Setup of easy-dexnet python codebase
Author: Lai QE
"""
from setuptools import setup,find_packages

requirements = [
      'opencv-python' 
]

setup(name='easy-gqcnn',                       
      version='0.0.1',                         
      description='easy-gqcnn project',
      author='LaiQE',
      author_email='1311127080@qq.com',
      package_dir = {'': 'src'},                
      packages=find_packages('src'),            
      install_requires=requirements,            
      # test_suite='test'
     )

