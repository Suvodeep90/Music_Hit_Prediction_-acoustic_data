#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 18:33:06 2018

@author: Amanul

This code uses spotify api (spotipy) to collect elements of music from the spotify databse. The api takes artist name as input and
returns attributes such as - accountiness, danceability, loudness, speechiness, acousticness, instrumentalness, liveness and so on.
Running the code - 
Once the code starts execution it opens up spotify login page that you need to login to to get access to the database, once logged
in you need to copy the url and paste on ur console for the exection to proceed. Once the attributes are fetch they are automatically
stored a csv file at the specifid data location.
You may need to create a developer account at spotify to get a client_id and client_password to access (the one included in the code is 
my client Id and password and you may not be able to use it because you would need my spotify credentials to login when the brownser prompts
"""

from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import spotipy
import numpy as np
import pandas as pd
import json
import spotipy
import time
import sys
import csv
import time
import os

username='hit_factory'
os.remove(f".cache-{username}")
token = util.prompt_for_user_token(username,client_id='386178169c234d699eaad5b82e22b5c7',client_secret='45cd0d802f0d4ecda69ccd7f6dc52634',redirect_uri='https://www.google.com/')
sp = spotipy.Spotify(auth=token)

file = 'data.csv'
data = pd.read_csv(file, encoding = 'utf-8')

artist = np.array(data.artist)
tid = []
tname = []

output_file = 'spotify_data.csv'

artists = set('Future')
for art in artists:
    results = sp.search(q=art, limit=2)
    for i, t in enumerate(results['tracks']['items']):
        tname.append(t['name'])
        tid.append(t['uri'])

out = open(output_file, 'w', newline='', encoding='utf8')
fieldnames = ['id','track','danceability', 'energy','key','loudness', 'mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
writer = csv.DictWriter(out,fieldnames)
writer.writerow({'id':'id','track':'track','danceability':'danceability', 'energy':'energy','key':'key','loudness':'loudness', 'mode':'mode','speechiness':'speechiness','acousticness':'acousticness','instrumentalness':'instrumentalness','liveness':'liveness','valence':'valence','tempo':'tempo','duration_ms':'duration_ms','time_signature':'time_signature'})            
        
i = 0
for uri in tid:
    features = sp.audio_features(uri)
    dic = features[0]
    writer.writerow({'id': dic['id'], 'track':tname[i],'danceability':dic['danceability'], 'energy':dic['energy'],'key':dic['key'],'loudness':dic['loudness'], 'mode':dic['mode'],'speechiness':dic['speechiness'],'acousticness':dic['acousticness'],'instrumentalness':dic['instrumentalness'],'liveness':dic['liveness'],'valence':dic['valence'],'tempo':dic['tempo'],'duration_ms':dic['duration_ms'],'time_signature':dic['time_signature']}) 
    i+=1
    print("successfully done")
    
out.close()
