# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:52:32 2018

@author: suvod
"""

import csv
import pandas as pd

df = pd.read_csv('C:\\Users\\suvod\ALDA\\song_dataset.csv')

songName = df['track'].tolist()
#print(songName)

rest = []
sep = '('
j = 0
for i in songName:
    rest.append('http://www.officialcharts.com/search/singles/' + (str(i).split(sep,1)[0]).replace(" ", "%20") + '/')
    j += 1
    
    
df['link'] = pd.DataFrame(data = rest)
print(df['link'])

df.to_csv('C:\\Users\\suvod\ALDA\\song_dataset_final.csv', encoding =  'utf-8')