# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:37:21 2018

@author: suvod
"""

from __future__ import print_function
import re
import requests
from bs4 import BeautifulSoup as bs
import csv
import pandas as pd

df = pd.read_csv('C:\\Users\\suvod\ALDA\\song_dataset_final.csv')
i = 0
print(df.shape)
links = df['link'].tolist()
print(links)
pos = []
woc = []

for link in links:
    try:
        page = requests.get(link)
        print(link)
        
        content = page.content
        
        # Create a BeautifulSoup object
        soup = bs(content, 'html.parser')
        
        
        table = soup.findChildren('table')[0]
        
        rows = table.findChildren('tr')
        
        j = 0
        for row in rows:
            cells = row.findChildren('td')
            i = 0
            for cell in cells:
                cell_content = cell.getText()
                clean_content = re.sub( '\s+', ' ', cell_content).strip()
                #print(clean_content)
                i = i + 1
                if "Sorry, there are no Official Singles Chart results" in clean_content:
                    pos.append('0')
                    woc.append('0')
                #if i in [5,4,3]:
                    #print(clean_content)
                if i == 3:
                    pos.append(clean_content)     
                elif i == 4:
                    woc.append(clean_content)
                
            #print('value of i:', i)
            j += 1
            #print('value of j:', j)
            if j >= 2:
                break
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
    except IndexError:
        continue
    except:
        break
#artist_name_list1 = soup.find(class_='chart-results-content')
#artist_name_list2 = soup.find
#artist_name_list_items = artist_name_list1
df['Peak_Pos'] = pd.DataFrame(data = pos)
df['WoC'] = pd.DataFrame(data = woc)
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.to_csv('C:\\Users\\suvod\ALDA\\song_dataset_data.csv', encoding =  'utf-8')

#print(artist_name_list1.prettify())
#
#for artist_name in artist_name_list_items:
#    names = artist_name.contents[0]
#    print(names)