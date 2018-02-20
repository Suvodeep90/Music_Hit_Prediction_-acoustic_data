# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:52:32 2018

@author: suvod
"""

import csv
import pandas as pd
import os

data_loc = 'data'
cwd = os.getcwd()
source_file = 'song_dataset.csv'
destination_file = 'song_dataset_url.csv'
data_path = os.path.join(cwd, data_loc)
source_file_path = os.path.join(data_path, source_file)
destination_file_path = os.path.join(data_path, destination_file)
df = pd.read_csv(source_file_path)

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

df.to_csv(destination_file_path, encoding =  'utf-8')