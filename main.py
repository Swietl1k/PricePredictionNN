import numpy as np
import pandas as pd
import data_processing, model_utils, train 


if __name__ == "__main__":
    db_path = "200k_cars.csv"
    df = pd.read_csv(db_path)
    
    cleaned_df = data_processing.clean_db(df)
    price_min = cleaned_df['price'].min()
    price_max = cleaned_df['price'].max()
    encoded_df = data_processing.data_encoding(cleaned_df)
    data_div = data_processing.divide_data(encoded_df, 0.2)
    
    X_train, Y_train = data_div['X_train'], data_div['Y_train']
    X_test, Y_test = data_div['X_test'], data_div['Y_test']
    
    m, n = X_train.shape
    params = model_utils.init_params(m)
    trained_params = train.train_loop(data_div, params, epochs=150, learning_rate=0.1, batch_size=16)

    test_cache = train.forward_pass(X_test, trained_params)
    Y_test_hat = test_cache['output']
    test_loss = model_utils.compute_loss(Y_test, Y_test_hat)
    print(f'Test Loss: {test_loss}')

    Y_test_hat_original = Y_test_hat * (price_max - price_min) + price_min
    Y_test_original = Y_test * (price_max - price_min) + price_min
    absolute_differences = np.abs(Y_test_hat_original - Y_test_original)
    average_difference = np.mean(absolute_differences)
    av_price = np.mean(Y_test_original)
    percent_diff = average_difference * 100 / av_price

    print(f"Average price difference (Mean Absolute Error): {average_difference:.2f}\n {percent_diff:.2f}%")

    
    
