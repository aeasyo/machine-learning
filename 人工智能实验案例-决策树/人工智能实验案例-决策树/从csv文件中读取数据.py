# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:11:05 2020

@author: mrliu
"""
import csv
dataSet = list()
f=open('E:/Data/11.csv') 
reader = csv.reader(f)
label = next(reader)
labels = label[:-1]
lines=f.readlines()
print(labels)
for line in lines:
    elements = line.strip().split(',') 
    dataSet.append(elements)
print(dataSet)