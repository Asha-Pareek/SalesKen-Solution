"""
Requirements : Levenshtein Distance Module
Command to Install : pip install python-Levenshtein
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

## Library to calculate Levenshtein distance
import Levenshtein as lv

def getIndex(bad_city, country):
    ## Initialize ID
    toReturn = np.nan

    ## Reduce the space of search using country
    temp = df_correct[df_correct.country == country]

    ## Since only 20% of the values are changed at max. [Reduces few comparisons]
    maxLivinglength = int(len(bad_city) * 0.2) + 1

    ## Initialize max distance
    highestLev = 100000

    for i in temp[['name', 'country', 'id']].values:
        lev_curr = lv.distance(bad_city, i[0])
        if highestLev > lev_curr:
            highestLev = lev_curr
            toReturn = i[2]
        else:
            continue
    return toReturn

## Read Csvs
df_correct = pd.read_csv('Correct_cities.csv')
df_incorr = pd.read_csv('Misspelt_cities.csv')


## Get the ids
df_incorr['id'] = df_incorr.apply(lambda x : getIndex(x['misspelt_name'], x['country']), axis = 1)


## Rename the columns 'id' to 'correct_city_id'
df_incorr.rename(columns = {'id': 'correct_city_id'}, inplace = True)


## Dump the file.
df_incorr.to_csv('missspeled_cities_corrected.csv', index = False)
