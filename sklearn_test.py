import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_processing import clean_db, data_encoding, divide_data


db_path = "../200k_cars.csv"
df = pd.read_csv(db_path)

cleaned_df = clean_db(df)
price_min = cleaned_df['price'].min()
price_max = cleaned_df['price'].max()
encoded_df = data_encoding(cleaned_df)

data_div = divide_data(encoded_df, 0.2)
X_train, Y_train = data_div['X_train'].T, data_div['Y_train'].flatten()
X_test, Y_test = data_div['X_test'].T, data_div['Y_test'].flatten()

model = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    learning_rate_init=0.0001,
    max_iter=1000,
    batch_size=16,
    random_state=42
)

model.fit(X_train, Y_train)

Y_test_hat = model.predict(X_test)

Y_test_hat_original = Y_test_hat * (price_max - price_min) + price_min
Y_test_original = Y_test * (price_max - price_min) + price_min

test_loss = mean_squared_error(Y_test_original, Y_test_hat_original)
absolute_differences = np.abs(Y_test_hat_original - Y_test_original)
average_difference = np.mean(absolute_differences)
av_price = np.mean(Y_test_original)
percent_diff = average_difference * 100 / av_price

print(f"Test Loss (MSE): {test_loss:.2f}")
print(f"Average price difference (Mean Absolute Error): {average_difference:.2f}")
print(f"Percentage difference: {percent_diff:.2f}%")
