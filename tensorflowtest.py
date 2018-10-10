# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:48:51 2018

@author: fmdu
"""


pi = 1.

for i in range (1, 10000):
    pi *= 4 * i ** 2/ (4*i**2 - 1.)
    
pi *= 2
print pi