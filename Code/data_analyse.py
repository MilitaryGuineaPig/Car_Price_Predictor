import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


raw_data = pd.read_csv("res/originalData.csv")

print(raw_data.head())
print(raw_data.columns)
# print(raw_data.count())
# print(raw_data.info())

# descriptive statistics summary
# print(raw_data['Price'].describe())

# Distribution
sns.displot(raw_data['Price'])
# plt.show()

# Total Missing Value
missing_values = raw_data.isnull().sum()
# print('Total Missing Value :\n',missing_values)

# Check the missing value in percentage
missing_values_percentage = (raw_data.isnull().sum() / len(raw_data)) * 100
print('\n\nMissing Value Percentage\n',missing_values_percentage)

