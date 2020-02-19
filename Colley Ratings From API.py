# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:35:10 2020

@author: Nate Z
This is the pure code accompanying the jupyter notebook outlining Colley's
Matrix Method. Unformatted and uncommented.
"""
import pandas as pd
import requests
import numpy as np

year = 2006
#Get from api
response = requests.get(r'https://api.collegefootballdata.com/games?'
                            'year={year}&seasonType=both'.format(year = year))
games = pd.read_json(response.text)

#FBS Games, played games, simplify
games = games[(~games['home_conference'].isnull()) & (~games['away_conference'].isnull())]
games = games[(games['home_points'] > 0) | (games['away_points'] > 0)]
games = games[['home_team','home_points','away_team','away_points']]

#home win boolean and ones column
games['home_win'] = -1+ 2*(games['home_points'] > games['away_points']).astype(int)
games['ones'] = 1

#team names
teams = pd.DataFrame(games['home_team'].append(games['away_team']).unique(),columns = ['team'])
teams = teams.sort_values(by = ['team']).reset_index(drop = True)


colley_vec = 1+(games[['home_team','home_win']].groupby('home_team').sum()\
         -games[['away_team','home_win']].groupby('away_team').sum())/2

games_played = (games[['home_team','ones']].groupby('home_team').sum()\
         +games[['away_team','ones']].groupby('away_team').sum())

diag = pd.DataFrame(2*np.identity(len(colley_vec))+np.diag(games_played['ones']),teams['team'],teams['team'])

piv1 = pd.pivot_table(games,values = 'ones',index = 'home_team', \
                      columns = 'away_team', aggfunc = np.sum).fillna(0)

piv2 = pd.pivot_table(games,values = 'ones',index = 'away_team', \
                      columns = 'home_team', aggfunc = np.sum).fillna(0)
    
colley_mat = diag - piv1 - piv2

colley_inv = pd.DataFrame(np.linalg.pinv(colley_mat.values),colley_mat.columns,colley_mat.index)

ratings = colley_inv.dot(colley_vec)
ratings = ratings.rename(columns={'home_win':'rating'})
ratings = ratings.sort_values(by = ['rating'], ascending = False)
print(ratings.head(10))