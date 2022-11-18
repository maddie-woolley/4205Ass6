# part 1-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# imports data set and prints the main data, as well as the description of dataset
crop_prediction = pd.read_csv('cpdata (1).csv')
# print(crop_prediction)
# print(crop_prediction.describe())
# check for nulls
# print(crop_prediction.isnull())
# outputs no nulls


# Part 2-
# initialize/ print correlation
import seaborn as sb

corrMatrix = crop_prediction.corr()
# print(corrMatrix)
# sb.heatmap(corrMatrix, annot=True)
# plt.show()


# Part 3-
# count-plot
sb.countplot(data=crop_prediction, y="label")
plt.show()
# Box Plot
sb.boxplot(crop_prediction[['temperature', 'humidity', 'ph', 'rainfall']])
plt.show()
# pairplot
sb.pairplot(crop_prediction)
plt.show()

# Part 4 -Random Forest


# Part 4 - KNN

# Part 4 - SWM
