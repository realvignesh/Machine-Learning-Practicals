import numpy as np

# Features X: [Relative Strength Index (RSI), Dividend Yield]
X = np.array(([2, 9], [1, 5], [3, 6], [4, 4], [7, 8], [5, 8], [8, 9], [6, 7], [7, 9], [10, 12]), dtype=float)
# Target Y: Price
y = np.array(([92], [86], [89], [87], [94], [87], [96], [90], [91], [88]), dtype=float)
X = X / np.amax(X, axis=0)  # Maximum of X array longitudinally
y = y / 100


# SIGMOID FUNCTION
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# DERIVATIVE OF SIGMOID FUNCTION
def derivatives_sigmoid(x):
    return x * (1 - x)


# VARIABLE INITIALIZATION
epoch = 5000  # Setting training iterations
lr = 0.1  # Setting learning rate
input_layer_neurons = 2  # Number of features in data set
hidden_layer_neurons = 3  # Number of hidden layers neurons
output_neurons = 1  # Number of neurons at output layer

# WEIGHT AND BIAS INITIALIZATION
# Draws a random range of numbers uniformly of dim x*y
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wout = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))


output = None
for i in range(epoch):
    # FORWARD PROPAGATION
    hinp1 = np.dot(X, wh)
    hinp = hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1 = np.dot(hlayer_act, wout)
    outinp = outinp1 + bout
    output = sigmoid(outinp)

    # BACKPROPAGATION
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)

    # HOW MUCH HIDDEN LAYER wts CONTRIBUTED TO THE ERROR
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
    wout += hlayer_act.T.dot(d_output) * lr

    # DOTPRODUCT OF NEXTLAYERERROR AND CURRENTLAYEROP
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X.T.dot(d_hiddenlayer) * lr

print("Normalized Training Examples: \n" + str(X))
print("\nActual Prices: \n" + str(y))
print("\nPredicted Prices: \n", output)
