# deeplearning-multimodal-neural-network
Reproducing work from https://medium.com/@dave.cote.msc/hybrid-multimodal-neural-network-architecture-combination-of-tabular-textual-and-image-inputs-7460a4f82a2e

Other sources: 
- Dataset: https://www.kaggle.com/datasets/ericpierce/austinhousingprices
- Dataset: https://data.texas.gov/Business-and-Economy/key-economic-indicators/np6r-w7eh
- Kaggle API: https://www.kaggle.com/docs/api

This is an on-going work performed on Github Codespaces (it's awesome).

Logbook:
1. Preprocessed tabular data
- File: src/preprocessing_tabular.py
- Demo: src/preprocessing_tabular.ipynb
- What's done: Drop NaN, removed outliers, adjusted (sale price) with inflation table (CPI)
- Todo: split train,test,validate sets

2. Preprocessing image data
- Plan: Deploy/Transfer learn a model for classifying images to: is a house, not a house
- File: 
- Demo: