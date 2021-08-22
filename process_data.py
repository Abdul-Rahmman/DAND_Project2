# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

messages = pd.read_csv('messages.csv')
print(messages.head(4))

categories = pd.read_csv('categories.csv')
print(categories.head(4))

print(messages.columns)
print(categories.columns)