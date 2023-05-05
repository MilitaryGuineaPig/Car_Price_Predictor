import warnings
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.model_selection import train_test_split


# Split the data
def data_split(data):
    X = data.drop('Price', axis=1).values
    y = data['Price'].values

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print("Done data split")
    return X_train, X_test, y_train, y_test


def data_fill_empty(raw_data):
    # print(raw_data.columns)

    # --------- Part 1 - Model
    # print(raw_data['Model'].unique())
    # Lowercase all models
    raw_data['Model'] = raw_data['Model'].str.lower()
    # print(raw_data['Model'].unique())

    # --------- Part 2 - Kilometres
    # Check all the available year in the data
    unique_years = sorted(raw_data['Year'].unique())
    # print(unique_years)
    # print(raw_data['Kilometres'])
    # Remove the 'km' string from the column
    raw_data['Kilometres'] = raw_data['Kilometres'].str.replace(' km', '')
    # Transform the column to integer
    raw_data['Kilometres'] = pd.to_numeric(raw_data['Kilometres'], errors='coerce')

    # Fill the missing data with mean group by year
    raw_data['Kilometres'] = raw_data['Kilometres'].fillna(raw_data.groupby('Year')['Kilometres'].transform('mean'))

    # Check
    # print('Total Missing Value :\n', raw_data['Kilometres'].isnull().sum())
    # Drop the remaining missing values
    raw_data.dropna(subset=['Kilometres'], inplace=True)
    # Check how much have I dropped
    # print('Total Missing Value :\n', raw_data['Kilometres'].isnull().sum())

    # --------- Part 3 - Body type
    # Fill missing body type with same model
    raw_data['Body Type'] = raw_data.groupby('Model')['Body Type'].ffill().bfill()

    # --------- Part 4 - Engine & Cylinder
    # define a function to extract the number from the string
    def extract_num(s):
        match1 = re.search('(?<=[IVL])-?([0-9]+)(?=$|\s)', s)  # Extract number after the character V,L, or I
        match2 = re.search('(\d+)\s?-?(?i)\w*cyl\w*', s)  # Extract number before the word 'cyl'
        match3 = re.search('(?i).*electric.*', s)  # Fill electric cars cylinder with 0
        if match1:
            if int(match1.group(1)) <= 12:
                return int(match1.group(1))
        elif match2:
            if int(match2.group(1)) <= 12:
                return int(match2.group(1))
        elif match3:
            return 0
        else:
            return None

    raw_data[' Engine'] = raw_data[' Engine'].astype('str')
    raw_data['Cylinder'] = raw_data[' Engine'].apply(extract_num)


    numpy.set_printoptions(threshold=sys.maxsize)
    # print(raw_data['Cylinder'].unique())
    # print(raw_data[' Engine'].unique())
    # len(raw_data['Cylinder'].unique())

    # Check the 1cylinder car
    # print(raw_data[raw_data['Cylinder'] == 1])
    # Fix the data for Cayenne
    raw_data.at[6628, 'Cylinder'] = '6'
    # Check the 2cylinder car
    # print(raw_data[raw_data['Cylinder'] == 2])

    # Fill missing Cylinder type with same model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw_data['Cylinder'].fillna(raw_data.groupby('Model')['Cylinder'].transform('median'), inplace = True)

    # Fill the rest with the same body type
    raw_data['Cylinder'].fillna(raw_data.groupby('Body Type')['Cylinder'].transform('median'), inplace = True)
    raw_data['Cylinder'] = raw_data['Cylinder'].astype(int)
    # print('Total Missing Value :\n', raw_data['Cylinder'].isnull().sum())
    # print(raw_data['Cylinder'])

    # --------- Part 5 - Transmission
    # print(raw_data[" Transmission"])
    # Split the transmission column to speed and type
    raw_data[['Speed', 'Type']] = raw_data[' Transmission'].str.extract(r'^(\d+)?\s?(.*)')

    # Most electric car have 1 speed type
    raw_data['Speed'] = raw_data['Speed'].fillna(raw_data['Cylinder'].apply(lambda x: 1 if x==0 else None))
    # Fill with the same model, sorted by year
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw_data['Speed'] = raw_data['Speed'].fillna(raw_data.sort_values('Year').groupby('Model')['Speed'].ffill().bfill())
    raw_data['Speed'] = raw_data['Speed'].astype(int)
    raw_data['Type'] = raw_data['Type'].str.extract('(Automatic|Manual|Sequential|CVT)', expand=False)
    raw_data['Type'] = raw_data.groupby('Model')['Type'].ffill().bfill()

    # --------- Part 6 - Drivetrain
    # Fill missing Drive train type with same model
    raw_data[' Drivetrain'] = raw_data.groupby('Model')[ ' Drivetrain'].ffill().bfill()

    # --------- Part 7 - Exterior Colour
    def extract_main_color(color_str):
        main_color = ['white', 'black', 'blue', 'red', 'gray', 'silver', 'yellow', 'green', 'brown', 'orange', 'purple', 'gold']
        color_lower = color_str.lower()
        color_lower = re.sub(r'\b(thunder|quartz|metallescent|shadow|aluminum|grey|graphite|charcoal|ash|sterling|steel|gun|granite|magnetic|billet|metal)\b', r'gray', color_lower)
        color_lower = re.sub(r'\b(burgundy|maroon|scarlet|rosso|ruby|molten|flame)\b', r'red', color_lower)
        color_lower = re.sub(r'\b(bianco|chalk|powder|aspen|avalanche|polar|snow|cream|iridescent)\b', r'white', color_lower)
        color_lower = re.sub(r'\b(sea|area|earl|blu|wave|marine|frostbite)\b', r'blue', color_lower)
        color_lower = re.sub(r'\b(mango|tangerine|punk|sunset|papaya)\b', r'orange', color_lower)
        color_lower = re.sub(r'\b(dark|ebony|titanium|onyx|nero|rocky|midnight)\b', r'black', color_lower)
        color_lower = re.sub(r'\b(beige|dust|tan|copper|dusk|chestnut|desert|mahogany|nickel|palladium|birch)\b', r'brown', color_lower)
        color_lower = re.sub(r'\b(platinum|mist|frost|moon)\b', r'silver', color_lower)
        color_lower = re.sub(r'\b(velocity)\b', r'yellow', color_lower)
        color_lower = re.sub(r'\b(passion)\b', r'purple', color_lower)
        color_lower = re.sub(r'\b(olive|sublime)\b', r'green', color_lower)
        for color in main_color:
            if re.search(r'w*'+ color+ '\w*', color_lower):
                return color
        return None
    raw_data[' Exterior Colour'] = raw_data[' Exterior Colour'].astype(str)
    raw_data['Main Color'] = raw_data[' Exterior Colour'].apply(extract_main_color)

    raw_data['Main Color'].value_counts().head(30)
    # Fill missing Exterior Color type with same model
    raw_data['Main Color'] = raw_data.groupby('Model')[ 'Main Color'].ffill().bfill()

    # --------- Part 8 - Interior Colour
    raw_data[' Interior Colour'].value_counts()

    def extract_int_color(color_str):
        main_color = ['black','grey','red','white','blue','orange','cream']
        color_lower = color_str.lower()
        color_lower = re.sub(r'\b(stone|charcoal)\b', r'gray', color_lower)
        color_lower = re.sub(r'\b(burgundy)\b', r'red', color_lower)
        color_lower = re.sub(r'\b(taupe|beige|tan)\b', r'brown', color_lower)
        for color in main_color:
            if re.search(r'w*'+ color+ '\w*', color_lower):
                return color
        return None
    raw_data[' Interior Colour'] = raw_data[' Interior Colour'].astype(str)
    raw_data['Int Colour'] = raw_data[' Interior Colour'].apply(extract_int_color)
    # Fill missing Exterior Color type with same model
    raw_data['Int Colour'] = raw_data.groupby('Model')[ 'Int Colour'].ffill().bfill()


    # --------- Part 9 - Passangers
    # Drop Passenger Column
    raw_data = raw_data.drop(' Passengers', axis=1)

    # --------- Part 10 - Doors
    raw_data[' Doors'].unique()
    raw_data[' Doors'].value_counts()

    def extract_doors(s):
        match = re.search(r'\d+\.?\d*' , s)
        if match:
            return float(match.group())
        else:
            return np.nan
    raw_data[' Doors'] = raw_data[' Doors'].astype(str)
    raw_data['Doors'] = raw_data[' Doors'].apply(extract_doors)

    raw_data['Doors'].value_counts()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw_data['Doors'].fillna(raw_data.groupby('Model')['Doors'].transform('median'), inplace=True)

    # Fill the rest with the same body type
    raw_data['Doors'].fillna(raw_data.groupby('Body Type')['Doors'].transform('median'), inplace=True)
    raw_data['Doors'] = raw_data['Doors'].astype(int)

    # --------- Part 11 - Fuel Type
    raw_data[' Fuel Type'].value_counts()

    def extract_fuel(fueltype):
        fuel_type = ['hybrid', 'electric', 'gas', 'diesel']
        fuel_lower = fueltype.lower()
        fuel_lower = re.sub(r'\b(hybrid)\b', r'hybrid', fuel_lower)
        fuel_lower = re.sub(r'\b(unleaded|gasoline|gaseous|flexible|flex)\b', r'gas', fuel_lower)
        for fuel in fuel_type:
            if re.search(r'w*'+ fuel + '\w*', fuel_lower):
                return fuel
        return None
    raw_data[' Fuel Type'] = raw_data[' Fuel Type'].astype(str)
    raw_data['Fuel Type'] = raw_data[' Fuel Type'].apply(extract_fuel)
    raw_data['Fuel Type'].value_counts()

    # Fill missing Fuel type type with same model
    raw_data['Fuel Type'] = raw_data.groupby('Model')[ 'Fuel Type'].ffill().bfill()

    # --------- Part 12 - City and Highway
    def extract_first_float(s):
        match = re.search(r"[-+]?\d*\.\d+", s)
        if match:
            return float(match.group())
        else:
            return None

    raw_data[' City'] = raw_data[' City'].astype(str)
    raw_data[' Highway'] = raw_data[' Highway'].astype(str)
    raw_data['City'] = raw_data[' City'].apply(extract_first_float)
    raw_data['Highway'] = raw_data[' Highway'].apply(extract_first_float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw_data['City'].fillna(raw_data.groupby('Model')['City'].transform('median'), inplace = True)
        raw_data['Highway'].fillna(raw_data.groupby('Model')['Highway'].transform('median'), inplace = True)

    #Fill the rest with the same body type
    raw_data['City'].fillna(raw_data.groupby('Body Type')['City'].transform('median'), inplace = True)
    raw_data['Highway'].fillna(raw_data.groupby('Body Type')['Highway'].transform('median'), inplace = True)
    # print(raw_data[raw_data['City'].isnull()])

    #only 1 rows where its still missing, we can drop this
    raw_data = raw_data.dropna(subset=['City'])
    missing_values_percentage = (raw_data.isnull().sum() / len(raw_data)) * 100
    # print('\n\nMissing Value Percentage\n',missing_values_percentage)

    # --------- Filter (Drop Uneccesary Column)
    raw_data = raw_data.drop(['Unnamed: 0',' Engine',' Transmission',' Exterior Colour',' Interior Colour',' Doors',' Fuel Type',
                  ' City',' Highway'], axis=1)
    # print(raw_data.dtypes)

    print("fill done")
    return raw_data


def dat_out_del(df):
    # Set the 0.5% and 99.5% quantiles of the distribution
    cq005 = df['City'].quantile(0.005)
    cq995 = df['City'].quantile(0.995)
    print(cq005, cq995)
    hq005 = df['Highway'].quantile(0.005)
    hq995 = df['Highway'].quantile(0.995)
    print(hq005, hq995)

    # Drop the lowest 0.05% and highest 99.5%
    df = df[(df['City'] >= cq005) & (df['City'] <= cq995)]
    df = df[(df['Highway'] >= hq005) & (df['Highway'] <= hq995)]


    kmq005 = df['Kilometres'].quantile(0.005)
    kmq995 = df['Kilometres'].quantile(0.995)
    print(kmq005, kmq995)

    print('Percentage of new car :', df['Kilometres'].value_counts()[0] / len(df) * 100, '%')
    df = df[df['Kilometres'] > 0]
    km = np.array(df['Kilometres']).reshape(-1, 1)
    high_range = km[km[:, 0].argsort()][-10:]
    print('\nouter range (high) of the distribution:')
    print(high_range)
    df = df[df['Kilometres'] < 1000000]

    price = df['Price'].to_numpy()[:, np.newaxis]
    low_range = price[price[:, 0].argsort()][:10]
    high_range = price[price[:, 0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)

    # Set the 0.5% and 99.5% quantiles of the distribution
    pq005 = df['Price'].quantile(0.005)
    pq995 = df['Price'].quantile(0.995)
    print(pq005, pq995)

    # Drop the lowest 0.05% and highest 99.5%
    df = df[(df['Price'] >= pq005) & (df['Price'] <= pq995)]

    print("outer done")

    return df


def data_norma(df):
    # Log Transform
    df['Price'] = np.log(df['Price'])
    df['Kilometres'] = boxcox1p(df['Kilometres'], boxcox_normmax(df['Kilometres'] + 1))
    df['City'] = boxcox1p(df['City'], boxcox_normmax(df['City'] + 1))

    print("norm done")
    return df


def data_encod(df):
    df = pd.get_dummies(df).reset_index(drop=True)
    print("encod done")
    return df