import numpy as np 


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


def relu(x) -> np.ndarray:
    return np.where(x > 0, x, 0)

def relu_derivative(x, alpha=0.01) -> np.ndarray:
    dx = np.ones_like(x)
    dx[x < 0] = 0
    return dx

def leaky_relu(x, alpha=0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01) -> np.ndarray:
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


def compute_loss(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    m = Y.shape[1]  
    loss = (1/m) * np.sum((Y_hat - Y) ** 2)  # MSE
    return loss


def update_learning_rate(initial_lr, epoch, decay_rate=0.98, decay_steps=100):
    return initial_lr * decay_rate ** (epoch / decay_steps)
