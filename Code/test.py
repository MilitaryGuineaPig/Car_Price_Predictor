import pandas as pd
import pickle

from data_norm import data_fill_empty, data_encod, dat_out_del, data_norma
from model_build import model_fit


raw_data = pd.read_csv("res/originalData.csv")

data = data_fill_empty(raw_data)
data = dat_out_del(data)
data.to_csv('mod_data.csv', index=False)
data = data_norma(data)
data = data_encod(data)



"""# Fit the model
model = model_fit(data)
# Save the trained model to a file
with open("random_forest_model_2.pkl", 'wb') as file:
    pickle.dump(model, file)
"""
