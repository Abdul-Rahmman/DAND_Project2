# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:42:22 2021

@author: Manee44
"""
#Testing

from flask import Flask

app=Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"
