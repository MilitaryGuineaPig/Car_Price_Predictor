import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


def print_result(entries):
    # Get user data from the entries
    user_data = get_user_data(entries)
    # Create a dictionary with the user data and attribute names
    attributes = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level',
                  'blood_glucose_level']
    data_dict = dict(zip(attributes, user_data))

    # Load label encoders for gender and smoking history
    label_encoder_gender = joblib.load('encoder_gen.joblib')
    label_encoder_smoking_history = joblib.load('encoder_smok.joblib')
    # Transform gender and smoking history using the label encoders
    data_dict['gender'] = label_encoder_gender.transform([data_dict['gender']])
    data_dict['smoking_history'] = label_encoder_smoking_history.transform([data_dict['smoking_history']])

    # Convert dictionary values to appropriate types
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.size == 1:
            data_dict[key] = np.squeeze(value).item()
        else:
            data_dict[key] = value
    # Convert dictionary to a 2D numpy array
    input_array = np.array(list(data_dict.values()))
    input_array = input_array.reshape(1, -1)

    # Load the diabetes prediction model and predict the probability of diabetes
    diabetes_model = joblib.load('random_forest_model_94.pkl')
    result = diabetes_model.predict_proba(input_array)[0][1]
    formatted_result = "{:.1f}".format(result)
    # Clear the result entry and insert the formatted result
    result_entry.delete(0, tk.END)
    result_entry.insert(0, formatted_result)


def close_window(window):
    # Destroy the information window
    window.destroy()


def info_wind():
    # Create a new window for information
    info_window = tk.Toplevel(root)
    info_window.title("Info")
    # Add a label with information text to the new window
    information = "Car Price Predictor Program\n\n\n" \
                  "This program allows you to predict the estimated price of a car based on some information about the car. To use the program, please follow these steps:\n\n\
                  Enter the car's information in the fields provided. The fields are as follows:\n" \
                  "Brand: Enter the brand of the car (e.g. Toyota, Honda, Ford)\n" \
                  "Model: Enter the model of the car (e.g. Camry, Civic, Mustang)\n" \
                  "Year: Enter the year the car was made\n" \
                  "Mileage: Enter the current mileage of the car\n" \
                  "Transmission: Enter the type of transmission (e.g. manual, automatic)\n" \
                  "Fuel type: Enter the type of fuel the car uses (e.g. gasoline, diesel, electric)\n" \
                  "Condition: Enter the condition of the car (e.g. new, used, like-new)\n\n\
                  Please note that all fields are required.\n\n" \
                  "Once you have entered the car's information, click on the \"Result\" button. The program will " \
                  "calculate the estimated price of the car based on its information.\n\n" \
                  "The result will be displayed in the \"Result\" field, which shows the estimated price of the car.\n"\
                  "Please note that the car price prediction is based on a statistical model and is " \
                  "not a substitute for " \
                  "professional appraisal or market research. If you are considering buying or selling a car, it's " \
                  "recommended to consult with a qualified car appraiser or do additional market research."
    tk.Label(info_window, text=information, justify='left', wraplength=400).pack(padx=10, pady=10)
    # Add a close button to the new window
    close_button = tk.Button(info_window, text="Close", command=lambda: close_window(info_window))
    close_button.pack(padx=10, pady=10)


def get_user_data(entries):
    user_data = []
    # Get info from each line
    for entry in entries:
        value = entry.get()
        if value.isdigit():
            user_data.append(float(value))
        else:
            user_data.append(value)
    # Return an array
    return user_data


# Creating main window
root = tk.Tk()
root.title("Car Price Prediction")

# Creating lists of variants
data = pd.read_csv("mod_data.csv")
print(data.columns)
Make_values = data['Make'].unique().tolist()
Model_values = data['Model'].unique().tolist()
Body_values = data['Body Type'].unique().tolist()
Drivetrain_values = data[' Drivetrain'].unique().tolist()
Speed_values = data['Speed'].unique().tolist()
Type_values = data['Type'].unique().tolist()
Main_Color_values = data['Main Color'].unique().tolist()
Int_Colour_values = data['Int Colour'].unique().tolist()
Fuel_values = data['Fuel Type'].unique().tolist()
City_values = data['City'].unique().tolist()
Highway_values = data['Highway'].unique().tolist()


# create 15 label and entry widgets
labels = ['Year', 'Make', 'Model', 'Kilometres', 'Body Type', 'Drivetrain', 'Cylinder', 'Speed', 'Type', 'Main Color',
          'Int Colour', 'Doors', 'Fuel Type', 'City', 'Highway']

label_value_dict = {'Make': 'Make_values',
                    'Model': 'Model_values',
                    'Body Type': 'Body_values',
                    'Drivetrain': 'Drivetrain_values',
                    'Speed': 'Speed_values',
                    'Type': 'Type_values',
                    'Main Color': 'Main_Color_values',
                    'Int Colour': 'Int_Colour_values',
                    'Fuel Type': 'Fuel_values',
                    'City': 'City_values',
                    'Highway': 'Highway_values'}

entries = []
for i, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=10)
    if label in label_value_dict:
        values_var_name = label_value_dict[label]
        values = locals()[values_var_name]
        entry_var = tk.StringVar()
        entry_var.set(values[0])
        entry = ttk.Combobox(root, textvariable=entry_var, values=values)
    else:
        entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=10)
    entries.append(entry)


# Result field
result_label = tk.Label(root, text=labels[8])
result_label.grid(row=15, column=0, padx=10, pady=10)
result_entry = tk.Entry(root)
result_entry.grid(row=15, column=1, padx=10, pady=10)

# Create a button to print the result
result_button = tk.Button(root, text="Print result", command=lambda: print_result(entries))
result_button.grid(row=16, column=1, columnspan=1, padx=5, pady=5)

# Create a button to print the info
info_button = tk.Button(root, text="Info", command=info_wind)
info_button.grid(row=16, column=0, columnspan=1, padx=5, pady=5)

root.mainloop()