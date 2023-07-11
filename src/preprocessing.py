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

#Import tabular dataset
df_dataset = pd.read_csv('data/austinHousingData.csv')
print(df_dataset.head())
# X_train_structured_std = pd.read_csv('/data/X_train_structured.csv', sep = ';')
# X_test_structured_std = pd.read_csv('/data/X_test_structured.csv', sep = ';')