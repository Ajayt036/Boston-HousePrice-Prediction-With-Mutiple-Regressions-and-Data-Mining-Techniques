# Boston-Housing-Prediction-using-Regressions

## Summary:
The following case study of Boston Housing reads the historical data set from BostonHousing.csv, clean it, and then predict the median value of the House in Boston for new records using Multiple Regression Model. First it uses 13 predictors from the datasets for the target output. And then, it optimizes the accuracy of prediction of price by iterating number of predictors in the model using Exhaustive search method and forward selection of predictors in the dataset.

## Main Chapters:

1. Upload, explore, clean, and preprocess data for multiple linear regression.
2. Develop multiple linear regression with all 13 predictors.
3. Develop multiple linear regression with reduced number of predictors.

## Featuring:

1. Exhautive Search Method
2. Forward Selection Method

## Library Used:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

## Tool Used:

1. Ananocda: Jupyter Notebook


