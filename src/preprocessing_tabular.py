#######################
## Catherine Chia
## 
#######################
'''
Reproducing https://medium.com/@dave.cote.msc/hybrid-multimodal-neural-network-architecture-combination-of-tabular-textual-and-image-inputs-7460a4f82a2e
This script inspects the datasets (tabular and images)
'''

#Libraries
import numpy as np
import pandas as pd
import os 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats

#Definitions
def statistic(x):
    return stats.shapiro(x).pvalue


####################################################################
#Import tabular dataset
df_dataset = pd.read_csv('data/austinHousingData.csv')
print(df_dataset.shape)
print(df_dataset.head(3))
print(list(df_dataset.columns), )

####################################################################
#Basic data preprocessing
df_processed=df_dataset.copy()

#Check dimension 
print(df_processed.shape)

#Check datatype for each columns
print(df_processed.dtypes)

#Check which column has null value
nullseries = df_processed.isnull().sum()
print(nullseries[nullseries > 0])

# - 'numOfBathrooms': if zero and if 'yearBuilt' is earlier than 1989 inclusive, then become 1; if zero and if 'yearBuilt' is later than 1989, then become 2
df_processed.loc[(df_processed['numOfBathrooms']==0)& (df_processed['yearBuilt'] > 1989), 'numOfBathrooms'] = 2
df_processed.loc[(df_processed['numOfBathrooms']==0)& (df_processed['yearBuilt'] <= 1989), 'numOfBathrooms'] = 1

# - 'numOfBedrooms': if zero and if 'yearBuilt' is earlier than 1989 inclusive, then become 2; if zero and if 'yearBuilt' is later than 1989, then become 2.5
df_processed.loc[(df_processed['numOfBedrooms']==0)& (df_processed['yearBuilt'] > 1989), 'numOfBedrooms'] = 2
df_processed.loc[(df_processed['numOfBedrooms']==0)& (df_processed['yearBuilt'] <= 1989), 'numOfBedrooms'] = 1

# - 'garageSpaces': if more than 3, then become 3
df_processed.loc[(df_processed['garageSpaces']> 3), 'garageSpaces'] = 3

# - 'parkingSpaces': if more than 3, then become 3
df_processed.loc[(df_processed['parkingSpaces']> 3), 'parkingSpaces'] = 3

# - 'lotSizeSqFt': delete outliers (>IQR * 1.6, < IQR * 1.6)
# Computing IQR
Q1 = df_processed['lotSizeSqFt'].quantile(0.25)
Q3 = df_processed['lotSizeSqFt'].quantile(0.75)
IQR = Q3 - Q1
# delete outliers
df_processed = df_processed.query('(@Q1 - 1.5 * @IQR) <= lotSizeSqFt <= (@Q3 + 1.5 * @IQR)')

#Check dimension 
print("After processing lotSizeSqFt: " + str(df_processed.shape))

# - 'livingAreaSqFt': delete outliers (>IQR * 1.6, < IQR * 1.6)
# Computing IQR
Q1 = df_processed['livingAreaSqFt'].quantile(0.25)
Q3 = df_processed['livingAreaSqFt'].quantile(0.75)
IQR = Q3 - Q1
# delete outliers
df_processed = df_processed.query('(@Q1 - 1.5 * @IQR) <= livingAreaSqFt <= (@Q3 + 1.5 * @IQR)')

#Check dimension 
print("After processing livingAreaSqFt: " + str(df_processed.shape))

# - 'latitude': delete outliers (< 30.12)
# delete outliers
df_processed = df_processed.query('latitude > 30.12')

#Check dimension 
print("After processing latitude: " + str(df_processed.shape))

# - 'zipcode': delete non-Austin zipcode
# - 'numOfAccessibilityFeatures': Two-class binary (No: 0; Few: 1)
df_processed.loc[df_processed["numOfAccessibilityFeatures"] > 0, "numOfAccessibilityFeatures"] = 1

# - 'numOfPatioAndPorchFeatures': Two-class binary (No: 0; Few: 1)
df_processed.loc[df_processed["numOfPatioAndPorchFeatures"] > 0, "numOfPatioAndPorchFeatures"] = 1

# - 'numOfSecurityFeatures': Two-class binary (No: 0; Few: 1)
df_processed.loc[df_processed["numOfSecurityFeatures"] > 0, "numOfSecurityFeatures"] = 1

# - 'numOfWaterfrontFeatures': Two-class binary (No: 0; Few: 1)
df_processed.loc[df_processed["numOfWaterfrontFeatures"] > 0, "numOfWaterfrontFeatures"] = 1

# - 'numOfWindowFeatures': Two-class binary (No: 0; Few: 1)
df_processed.loc[df_processed["numOfWindowFeatures"] > 0, "numOfWindowFeatures"] = 1

# - 'numOfCommunityFeatures': Two-class binary (No: 0; Few: 1)
df_processed.loc[df_processed["numOfCommunityFeatures"] > 0, "numOfCommunityFeatures"] = 1

####################################################################
#Clustering longitude and latitude data
kmeans = KMeans(n_clusters=25, random_state=0, n_init='auto').fit(df_processed[['longitude', 'latitude']])
df_processed['regionCluster'] = kmeans.predict(df_processed[['longitude', 'latitude']])

#Visualize clusters
fig, ax = plt.subplots()
p_scatter = ax.scatter(df_processed['longitude'], df_processed['latitude'], c=df_processed['regionCluster'].astype(float), s=0.4)
plt.title("KMean-clustered house coordinates in Austin, Texas")
plt.xlabel("Longitude (deg)")
plt.ylabel("Latitude (deg)")
ax.legend(*p_scatter.legend_elements(), title='clusters')

plt.show()

####################################################################
#Target inflation adjustment
#Import tabular dataset and a brief inspection
df_CPI = pd.read_csv('data/key_economic_indicators.csv')
print(df_CPI.shape)
print(list(df_CPI.columns), )

#Dropna for empty CPI rows
df_CPI = df_CPI.dropna(subset=['Consumer Price Index TX'], axis=0)
print(df_CPI.shape)

#Pivot table to get average CPI per year
df_CPI_pivot = pd.pivot_table(df_CPI, values='Consumer Price Index TX', index=['Year'], columns=['Month'])

##Normality test
#The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.

l_normality = []
for index, row in df_CPI_pivot.iterrows():
    l_normality.append(statistic(row))

#Check if there is at least one element with significant pvalue (reject null hypothesis of being normally distributed)
#Apply median as average if not all rows are normally distributed, otherwise use mean
alpha = 0.05
if all(num > alpha for num in l_normality): #greater than alpha = not significant
    df_CPI_pivot['ave_CPI'] = df_CPI_pivot.mean(axis=1)
else: 
    df_CPI_pivot['ave_CPI'] = df_CPI_pivot.median(axis=1)

#Drop unused columns and made it a dictionary
df_CPI_ave = df_CPI_pivot[['ave_CPI']]
d_CPI_ave = df_CPI_ave['ave_CPI'].to_dict()


##Inflation adjustment
#Check for the sale year range
print("oldest house sale year: ", str(min(df_processed["latest_saleyear"])))
print("latest house sale year: ", str(max(df_processed["latest_saleyear"])))

#loop through house sales dataframe rows and adjust for inflation
df_processed['adjPrice']=1 
for ind, row in df_processed.iterrows(): #row here is disconnected to the original dataframe
    temp_CPI = d_CPI_ave[row['latest_saleyear']]
    df_processed.loc[ind, 'adjPrice'] = row['latestPrice']/temp_CPI * 100
