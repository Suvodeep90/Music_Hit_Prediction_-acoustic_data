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
* SVM
* Naive Bayes
* Logistic Regression
* Decision Trees
* Random Forests

We also implemented the following algorithms from scratch:
* Neural Network
* Naive Bayes
* Multilayer, tree based Naive Bayes using error correction

### Problems Faced ###
The primary issue faced in this project was that of a class imbalance. Intuitively, there are less hit songs than non-hit songs. To combat
this issue, we implemented the SMOTE algorithm to sample the minority class (hit songs) further. 

## Instructions ##
1. Using main.py, choose the settings and algorithm to perform hit song prediction. These same instructions are listed in main.py.
2. For "algo", select either "RF" for random forest or "DT" for decision tree for the recursive feature selection setting.
3. For "feature_selector", select among "consistency_subset", "selectkBest", or "recursive_feature_selector" to reduce the dimensionality
of the dataset by choosing the most relevant features.
4. When setting the******** data pre-processing, the parameters are "file_path", "data_path", "algo", "feature_selector", and three booleans. 
The booleans are whether to use normalization, missing value treatment, and if feature selection will be performed, repsectively.
Note that when "NN" is chosen, we achieved the best results when no feature selection is performed (i.e., the last boolean is False).
5. Choose the cross-validation folds for performance assessment using "nFold".
6. Finally, choose the type of model to run using "_learner". The following are all options:
    1) Naive Bayes: NB
    2) Decision Tree: DT
    3) Random Forest: RF
    4) Support Vector Machine: SVM
    5) Logistic Regression: LR
    6) Neural Network: MLP
    7) Our Implementation of Neural Network: NN
    8) AdaBoost: ADA
    9) Tree Based Naive Bayes with depth 2: NBL2
    10) Tree Based Naive Bayes with depth 3: NBL3
    11) Our Implementation of Naive Bayes: NBO

## Resources ##
Dorien Herremans, David Martens & Kenneth Sörensen (2014) Dance Hit Song Prediction, Journal of New Music Research, 43:3, 291-302.
