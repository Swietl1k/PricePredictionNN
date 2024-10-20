import numpy as np
import pandas as pd

def cleanDb(df: pd.DataFrame) -> pd.DataFrame:
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

def dataEncoding(df: pd.DataFrame) -> pd.DataFrame:
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


def init_params(n: int) -> dict[str, np.ndarray]:
    params = {}

    # Layer 1: 16 neurons, input size n_input
    params['W1'] = np.random.randn(16, n) * 0.01
    params['b1'] = np.zeros((16, 1))

    # Layer 2: 8 neurons, input size 16
    params['W2'] = np.random.randn(8, 16) * 0.01
    params['b2'] = np.zeros((8, 1))

    # Output layer: 1 neuron, input size 8
    params['W3'] = np.random.randn(1, 8) * 0.01
    params['b3'] = np.zeros((1, 1))

    return params


def leaky_relu(x, alpha=0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01) -> np.ndarray:
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx

    
def forward_pass(X: np.ndarray, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']
    
    Z1 = np.dot(W1, X) + b1
    A1 = leaky_relu(Z1)
    #print(Z1.shape)
    
    Z2 = np.dot(W2, A1) + b2
    A2 = leaky_relu(Z2)
    #print(Z2.shape)
    
    Z3 = np.dot(W3, A2) + b3
    output = Z3
    #print(Z3.shape)
    
    cache = {
        'Z1': Z1, 'A1': A1,
        'Z2': Z2, 'A2': A2,
        'Z3': Z3, 'output': output
    }
    
    return cache


def compute_loss(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    m = Y.shape[1]  
    loss = (1/m) * np.sum((Y_hat - Y) ** 2)  # MSE

    return loss


def update_learning_rate(initial_lr, epoch, decay_rate=0.98, decay_steps=100):
    return initial_lr * decay_rate ** (epoch / decay_steps)


def backprop(X: np.ndarray, Y: np.ndarray, cache: dict[str, np.ndarray], params: dict[str, np.ndarray], learning_rate: float) -> dict[str, np.ndarray]:
    m = X.shape[1]
    
    # Gradients for output layer
    dZ3 = cache['output'] - Y
    dW3 = (1/m) * np.dot(dZ3, cache['A2'].T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    
    # Gradients for layer 2
    dA2 = np.dot(params['W3'].T, dZ3)
    dZ2 = dA2 * leaky_relu_derivative(cache['Z2'])
    dW2 = (1/m) * np.dot(dZ2, cache['A1'].T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Gradients for layer 1
    dA1 = np.dot(params['W2'].T, dZ2)
    dZ1 = dA1 * leaky_relu_derivative(cache['Z1'])
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    params['W1'] -= learning_rate * dW1
    params['b1'] -= learning_rate * db1
    params['W2'] -= learning_rate * dW2
    params['b2'] -= learning_rate * db2
    params['W3'] -= learning_rate * dW3
    params['b3'] -= learning_rate * db3

    return params



def train(X_train: np.ndarray, Y_train: np.ndarray, params: dict[str, np.ndarray], epochs: int, learning_rate: float, batch_size: int = 16) -> dict[str, np.ndarray]:
    # batch_size of 32 has slower conversion but in the end its better because less overfitting  
    m = X_train.shape[1] 
    best_loss = float('inf')
    patience_counter = 0
    patience = 50

    for epoch in range(epochs):
        # Shuffle the data at the beginning of each epoch
        permutation = np.random.permutation(m)
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = Y_train[:, permutation]

        for i in range(0, m, batch_size):
            X_batch = X_train_shuffled[:, i:i + batch_size]
            Y_batch = Y_train_shuffled[:, i:i + batch_size]

            cache = forward_pass(X_batch, params)
            params = backprop(X_batch, Y_batch, cache, params, learning_rate)

        test_cache = forward_pass(X_test, params)
        test_loss = compute_loss(Y_test, test_cache['output'])

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}, test loss: {best_loss}")
                break

        # monitor loss in next epochs
        if epoch % 10 == 0:
            cache = forward_pass(X_train, params)
            epoch_loss = compute_loss(Y_train, cache['output'])
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')

        '''
        if epoch % 100 == 0:
            learning_rate = update_learning_rate(learning_rate, epoch)
            print(f'learning rate: {learning_rate}')
        '''

    return params




if __name__ == "__main__":
    db_path = "200k_cars.csv"
    df = pd.read_csv(db_path)
    
    cleaned_df = cleanDb(df)
    price_min = cleaned_df['price'].min()
    price_max = cleaned_df['price'].max()
    encoded_df = dataEncoding(cleaned_df)
    data_div = divide_data(encoded_df, 0.2)
    
    X_train, Y_train = data_div['X_train'], data_div['Y_train']
    X_test, Y_test = data_div['X_test'], data_div['Y_test']
    
    m, n = X_train.shape
    params = init_params(m)
    trained_params = train(X_train, Y_train, params, epochs=1500, learning_rate=0.1)

    test_cache = forward_pass(X_test, trained_params)
    Y_test_hat = test_cache['output']
    test_loss = compute_loss(Y_test, Y_test_hat)
    print(f'Test Loss: {test_loss}')

    Y_test_hat_original = Y_test_hat * (price_max - price_min) + price_min
    Y_test_original = Y_test * (price_max - price_min) + price_min
    absolute_differences = np.abs(Y_test_hat_original - Y_test_original)
    average_difference = np.mean(absolute_differences)
    av_price = np.mean(Y_test_original)
    percent_diff = average_difference / av_price

    print(f"Average price difference (Mean Absolute Error): {average_difference:.2f} \n {av_price:.2f} \n {percent_diff:.2f}")

    
    
