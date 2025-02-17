# 🚗 Car Price Neural Network
A neural network built from scratch, written in python using only numpy and pandas for convenience. The purpose of this model was to predict the price of used cars given their features.

# 📚 Data
For testing, I used 113,015 cars, each with a set of 22 different features. [US Used cars dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset) from Kaggle. 

# ⚡ Model Details
- **3-layer fully connected neural network**
  - Layer 1: 16 neurons, Leaky ReLU activation
  - Layer 2: 8 neurons, Leaky ReLU activation
  - Output layer: 1 neuron, linear output
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: Batch Gradient Descent with learning rate decay and early stopping
- **Training Parameters**:
  - Learning Rate: `0.1`
  - Batch Size: `16`
  - Train/Test Split: `80/20`

# ⚖️ Comparison
For comparison, I used an sklearn neural network. In my testing, the parameters below yielded the best results
- hidden_layer_sizes: `(16, 8)`
- activation: relu
- solver: adam
- learning_rate_init: `0.0001`
- batch_size: `16`

| Model          | Mean Absolute Error | MAE / avg(price)  |
|:---------------|:-------------:|:---------------:|
| My model       | 3050.16        | 10.83%          |
| Sklearn        | 3610.48       | 12.82%          |

# 🔮 Already tried improvements
Below are some tested techniques that showed negative or no improvements: 
- regularization techniques such as L2 and dropout
- activation functions: linear, ReLu

# 🚀 Further improvements
Below are techniques for me to implement and test in the future:
- Grid search

