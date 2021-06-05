import pandas as pd
import numpy as np
from sklearn import preprocessing
s= pd.read_csv("engine.csv")
s.head(2)

V = s['Love']
R = s['Average_Watch_time']
C = s['Average_Watch_time'].mean()
m = s['Love'].quantile(0.70)

#I selected 0.70 as my argument for quantile() to indicate that I was concerned only with videos that received at least as many like as 70% ofthe 
#VIDEOS of our dataset.
#Selecting our value for m is a bit arbitrary, so do try some experimentation here.

s['weighted_average'] = (V/(V+m) * R) + (m/(m+V) * C) #formula
s.head(2)

t = s.sort_values('weighted_average', ascending=False)
t.head(2)

# Recommender system
#recommendation based on scaled watch time ,Avg Watch Time, share, love & comments score

from sklearn import preprocessing
#The transformation is given by:
#X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#X_scaled = X_std * (max - min) + min

#MinMaxScaler subtracts the minimum value in the feature and then divides by the range


min_max_scaler = preprocessing.MinMaxScaler()
videosscaled = min_max_scaler.fit_transform(t[['weighted_average', 'Share','Comment','Spammy_Views']])

videop_norm = pd.DataFrame(videosscaled, columns=['weighted_average', 'Share','Comment','Spammy_Views'])
videop_norm.head(2)

#Getting the normalised data
s[['norm_weighted_average', 'norm_Share','norm_Comment','norm_Spammy_Views']] = videop_norm

#renaming the columns

s['score4'] = s['norm_weighted_average'] * 0.7  + s['norm_Share']*0.2 + s['norm_Comment']*0.1 - s['norm_Spammy_Views']*0.1
video4_scored = s.sort_values(['score4'], ascending= False)
video4_scored.head(2)
#Alogirthm of the Engine

videop_scored = s.sort_values('score4', ascending=False)

plt.figure(figsize=(25,4))

ax = sns.barplot(x=videop_scored['Creator_Name'].head(10), y=videop_scored['score4'].head(10), data=videop_scored, palette='bright')

#plt.xlim(3.55, 5.25)
plt.title('Top Recommended Videos-RE4', weight='bold')
plt.xlabel('Name', weight='bold')
plt.ylabel('Score4', weight='bold')
plt.savefig('video4_scored.png')
print('video4_scored')


#Plotting videos against the score

videop_scored.to_csv('RecommendationEng4')
