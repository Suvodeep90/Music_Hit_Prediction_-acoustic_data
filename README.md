# Music_Hit_Prediction_-acoustic_data
This project is for predicting if a song will be a hit song or not depending on the acoustic properties of the song.
Examples of the data features used in these predictions are energy, key, loudness, tempo, album, artist, among other properties.
The structure and methods of this work closely follow those of Herremans et al.

## Methods ##

### Data Gathering ###
* Using Spotify's API, we scraped the attributes of approximately 3000 songs. 
* We created labels of each song as a hit/non-hit by scraping an official charts website. Hit criteria can be changed, but we label a hit
song as one that reaches the top 100 in the official charts.

### Preprocessing ###
We used several techniques to handle:
* Missing input
* Deduplication of samples
* Normalization of features
* Feature selection for removal of correlated features

### Algorithms Used ###
The algorithms we used from built-in libraries are:
* SVM, Naive Bayes, Logistic Regression, Decision Trees, Random Forests

We also implemented the following algorithms from scratch:
* Neural Network
* Naive Bayes
* Multilayer, tree based Naive Bayes using error correction

### Problems Faced ###
The primary issue faced in this project was that of a class imbalance. Intuitively, there are less hit songs than non-hit songs. To combat
this issue, we implemented the SMOTE algorithm to sample the minority class (hit songs) further. 

## Resources ##
Dorien Herremans, David Martens & Kenneth Sörensen (2014) Dance Hit Song Prediction, Journal of New Music Research, 43:3, 291-302.
