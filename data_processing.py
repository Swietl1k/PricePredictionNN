import numpy as np 
import pandas as pd


def clean_db(df: pd.DataFrame) -> pd.DataFrame:
    # Remove incomplete data records and invalid values
    df.replace(["--", " "], np.nan, inplace=True)
    df.dropna(how='any', inplace=True)

    measurement_columns = ['height', 'length', 'width']
    for col in measurement_columns:
        df[col] = df[col].str.replace('in', '', regex=False).astype(float)

    df['fuel_tank_volume'] = df['fuel_tank_volume'].str.replace('gal', '', regex=False).astype(float)
    df['maximum_seating'] = df['maximum_seating'].str.replace('seats', '', regex=False).astype(int)
    df['torque'] = df['torque'].str.extract(r'(\d+)').astype(int)
    
    return df

def data_encoding(df: pd.DataFrame) -> pd.DataFrame:
    # Frequency encoding for data with high cardinality
    freq_enc_model = df['model_name'].value_counts()
    df['model_name_encoded'] = df['model_name'].map(freq_enc_model)
    #freq_enc_make = df['make_name'].value_counts()
    #df['make_name_encoded'] = df['make_name'].map(freq_enc_make)

    # One-hot encoding for values with less cardinality
    columns_one_hot = [
        'make_name',
        'body_type', 
        'engine_type', 
        'fuel_type', 
        'listing_color', 
        'transmission_display', 
        'wheel_system'
    ]
    
    df = pd.get_dummies(df, columns=columns_one_hot, drop_first=True)
    columns_to_drop = ['model_name']
    df.drop(columns=columns_to_drop, inplace=True)

    numerical_columns = ['daysonmarket', 'engine_displacement', 'fuel_tank_volume', 
                         'height', 'horsepower', 'length', 'maximum_seating', 'mileage', 
                         'price', 'torque', 'width', 'year', 'model_name_encoded']

    # min-max scaling
    for col in numerical_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)

    bool_columns = df.select_dtypes(include='bool').columns 
    df[bool_columns] = df[bool_columns].astype(int)

    return df


def divide_data(encoded_df: pd.DataFrame, test_scale: float) -> dict[str, np.ndarray]:
    m = encoded_df.shape[0]
    m_test = int(m * test_scale)
    data_div = {}

    data_Y = np.array(encoded_df['price']).reshape(1, -1)  
    encoded_df.drop(columns=['price'], inplace=True)
    data_X = np.array(encoded_df).T 

    data_div['Y_test'] = data_Y[:, 0:m_test]
    data_div['X_test'] = data_X[:, 0:m_test]

    data_div['Y_train'] = data_Y[:, m_test:m]
    data_div['X_train'] = data_X[:, m_test:m]

    return data_div