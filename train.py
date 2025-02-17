import numpy as np
import model_utils


def forward_pass(X: np.ndarray, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']
    
    Z1 = np.dot(W1, X) + b1
    A1 = model_utils.leaky_relu(Z1)
    #print(Z1.shape)
    
    Z2 = np.dot(W2, A1) + b2
    A2 = model_utils.leaky_relu(Z2)
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



def back_prop(X: np.ndarray, Y: np.ndarray, cache: dict[str, np.ndarray], params: dict[str, np.ndarray], learning_rate: float) -> dict[str, np.ndarray]:
    m = X.shape[1]
    
    # Gradients for output layer
    dZ3 = cache['output'] - Y
    dW3 = (1/m) * np.dot(dZ3, cache['A2'].T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    
    # Gradients for layer 2
    dA2 = np.dot(params['W3'].T, dZ3)
    dZ2 = dA2 * model_utils.leaky_relu_derivative(cache['Z2'])
    dW2 = (1/m) * np.dot(dZ2, cache['A1'].T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Gradients for layer 1
    dA1 = np.dot(params['W2'].T, dZ2)
    dZ1 = dA1 * model_utils.leaky_relu_derivative(cache['Z1'])
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    params['W1'] -= learning_rate * dW1
    params['b1'] -= learning_rate * db1
    params['W2'] -= learning_rate * dW2
    params['b2'] -= learning_rate * db2
    params['W3'] -= learning_rate * dW3
    params['b3'] -= learning_rate * db3

    return params



def train_loop(data_div: dict[str, np.ndarray], params: dict[str, np.ndarray], epochs: int, learning_rate: float, batch_size: int = 16) -> dict[str, np.ndarray]:
    X_train, Y_train = data_div['X_train'], data_div['Y_train']
    X_test, Y_test = data_div['X_test'], data_div['Y_test']
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
            params = back_prop(X_batch, Y_batch, cache, params, learning_rate)

        test_cache = forward_pass(X_test, params)
        test_loss = model_utils.compute_loss(Y_test, test_cache['output'])

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
            epoch_loss = model_utils.compute_loss(Y_train, cache['output'])
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')

        '''
        if epoch % 100 == 0:
            learning_rate = update_learning_rate(learning_rate, epoch)
            print(f'learning rate: {learning_rate}')
        '''

    return params