###############################################################################
# General Information
###############################################################################
# Author: ZhongYi Jiang

# main.py: From here, launch deep symbolic regression tasks. All
# hyperparameters are exposed (info on them can be found in train.py). Unless
# you'd like to impose new constraints / make significant modifications,
# modifying this file (and specifically the get_data function) is likely all
# you need to do for a new symbolic regression task.

###############################################################################
# Dependencies
###############################################################################

from DSRtrain import train
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

###############################################################################
# Main Function
###############################################################################

# A note on operators:
# Available operators are: '*', '+', '-', '/', '^', 'sin', 'cos', 'tan',
#   'sqrt', 'square', and 'c.' You may also include constant floats, but they
#   must be strings. For variable operators, you must use the prefix var_.
#   Variable should be passed in the order that they appear in your data, i.e.
#   if your input data is structued [[x1, y1] ... [[xn, yn]] with outputs
#   [z1 ... zn], then var_x should precede var_y.

def main():
    # Load training and test data
    X_constants, X_rnn, y_constants, y_rnn = get_data()

    # Perform the regression task
    results = train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        operator_list = ['*', '+', '-', '/', 'cos', 'sin', 'var_x','ln', 'exp'],
        min_length = 2,
        max_length = 20,
        type = 'lstm',
        num_layers = 2,
        hidden_size = 250,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'adam',
        inner_optimizer = 'rmsprop',
        inner_lr = 0.1,
        inner_num_epochs = 25,
        entropy_coefficient = 0.005,
        risk_factor = 0.95,
        initial_batch_size = 2000,
        scale_initial_risk = True,
        batch_size = 1000,
        num_batches = 500,
        use_gpu = False,
        apply_constraint = True,
        live_print = True,
        summary_print = True
    )

    # Unpack results
    epoch_best_rewards = results[0]
    epoch_best_expressions = results[1]
    best_reward = results[2]
    best_expression = results[3]

    # Plot best rewards each epoch
    plt.plot([i+1 for i in range(len(epoch_best_rewards))], epoch_best_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.title('Reward over Time')
    plt.show()

###############################################################################
# Getting Data
###############################################################################

def get_data():
    """Constructs data for model
    """
    X = np.random.uniform(-1,1,20)
    # X = np.arange(-1,1,0.1)
    y = X ** 3 + X ** 2 + X # Nguyen-1
    # y = X ** 4 + X ** 3 + X ** 2 + X  # Nguyen-2
    # y = X**5 + X**4 + X ** 3 + X ** 2 + X  # Nguyen-3
    # y = X**6 + X**5 + X**4 +X ** 3 + X ** 2 + X  # Nguyen-4
    # y = np.sin(X**2) * np.cos(X) - 1 # Nguyen-5
    # y = np.sin(X) + np.sin(X + X**2) # Nguyen-6
    # y = np.log(X + 1) + np.log(X**2 + 1) # Nguyen-7
    # y = np.sqrt(X) # Nguyen-8
    # y = 3.93*X**3 + 2.21*X**2 + 1.78*X # Nguyen-1c
    # y = np.sin(X**2)*np.cos(X)-0.75 # Nguyen-5c
    # y = np.log(X + 1.4) + np.log(X**2 + 1.3) # Nguyen-7c
    # y = np.sqrt(1.23*X) # Nguyen-8c
    X = X.reshape(X.shape[0], 1)

    # X = np.arange(0, 2, 0.05).reshape(40, 1).repeat(2, axis = 1)
    # X = np.random.uniform(0, 1, size=(20,2))
    # y = np.sin(X[:,0]) + np.sin(X[:,1]**2) # Nguyen-9
    # y = 2 * np.sin(X[:, 0]) * np.cos(X[:, 1]) # Nguyen-10
    # y = X[:,0]**X[:,1] # Nguyen-11
    # y = X[:,0]**4 - X[:,0]**3 + 0.5*X[:,1]**2 - X[:,1] # Nguyen-12
    # y = np.sin(1.5*X[:,0])*np.cos(0.5*X[:,1]) # Nguyen-10c
    # y = np.exp(-np.pi**2 * 0.4 * X[:,1])*np.sin(np.pi*X[:,0])



    # Split randomly
    comb = list(zip(X, y))
    random.shuffle(comb)
    X, y = zip(*comb)


    # Proportion used to train constants versus benchmarking functions
    training_proportion = 0.2
    div = int(training_proportion*len(X))
    X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
    y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
    X_constants, X_rnn = torch.Tensor(X_constants), torch.Tensor(X_rnn)
    y_constants, y_rnn = torch.Tensor(y_constants), torch.Tensor(y_rnn)
    return X_constants, X_rnn, y_constants, y_rnn

if __name__== '__main__':
    main()
