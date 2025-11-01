"""A module with NAS controller-related code."""
import torch
import torch.nn.functional as F
import numpy as np
import tools
import scipy
from utils import Logger, mkdir_p
import os
import torch.nn as nn
from computational_tree import BinaryTree
import conf,setenv,initparameters
import random
import function as func
from sampler import BatchSampler
import time
import argparse
import sys,time
import random
import math
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='NAS')

parser.add_argument('--left', default=-1, type=float)
parser.add_argument('--right', default=1, type=float)
parser.add_argument('--epoch', default=2000, type=int)
parser.add_argument('--bs', default=1, type=int)
parser.add_argument('--greedy', default=0, type=float)
parser.add_argument('--random_step', default=0, type=float)
parser.add_argument('--ckpt', default='Dim5', type=str)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--dim', default=3, type=int)
parser.add_argument('--tree', default='depth1', type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--percentile', default=0.5, type=float)
parser.add_argument('--base', default=100, type=int)
parser.add_argument('--domainbs', default=1000, type=int)
parser.add_argument('--bdbs', default=1000, type=int)
args = parser.parse_args()


import timeout_decorator
from aTEAM.optim import NumpyFunctionInterface
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
from scipy.optimize.slsqp import fmin_slsqp as slsqp
from scipy.optimize import fmin_bfgs as bfgs
from DSRtrain import train
import matplotlib.pyplot as plt


X = np.load('robertson_x.npy')
X[:,1] = 1e4*X[:,1]
y = np.load('robertson_y.npy')[:,0]+ np.random.normal(loc=0, scale=0.006, size=3000)
y = y.reshape(3000,1)
print('X.shape:',X.shape)
print('y.shape:',y.shape)
# print('X[:10]',X[:10])
# print('y[:10]',y[:10])

pred = -0.04 * X[:,0] + 1 * X[:,1] * X[:,2]
pred = pred.reshape(3000,1)
# print('pred[:10]',pred[:10])
print('pred shape:', pred.shape)

# print('y:',y)
loss = nn.MSELoss()
val = torch.sqrt(loss(torch.tensor(pred), torch.tensor(y)))  # Convert to RMSE
val = torch.std(torch.tensor(y)) * val  # Normalize using stdev of targets
val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
# val = 1 / (1 + val)
print('true loss', val.item())
# time.sleep(10)

comb = list(zip(X, y))
random.shuffle(comb)
X, y= zip(*comb)

# Proportion used to train constants versus benchmarking functions
training_proportion = 0.0

div = int(training_proportion * len(X))
X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
X_constants, X_rnn = torch.Tensor(X_constants), torch.Tensor(X_rnn)
y_constants, y_rnn = torch.Tensor(y_constants), torch.Tensor(y_rnn)
y_constants = torch.flatten(y_constants)
y_rnn = torch.flatten(y_rnn)
print('X_constants:shape')
print(X_constants.shape)
print('y_constants shape')
print(y_constants.shape)

unary = func.unary_functions
binary = func.binary_functions
unary_functions_str = func.unary_functions_str
unary_functions_str_leaf = func.unary_functions_str_leaf
binary_functions_str = func.binary_functions_str

left = args.left
right = args.right
dim = args.dim

def get_boundary(num_pts, dim):

    bd_pts = (torch.rand(num_pts, dim)) * (args.right - args.left) + args.left

    num_half = num_pts//2
    xlst = torch.arange(0, num_half)
    ylst = torch.randint(0, dim, (num_half,))
    bd_pts[xlst, ylst] = args.left

    xlst = torch.arange(num_half, num_pts)
    ylst = torch.randint(0, dim, (num_half,))
    bd_pts[xlst, ylst] = args.right

    return bd_pts


class candidate(object):
    def __init__(self, action, expression, error):
        self.action = action
        self.expression = expression
        self.error = error

class SaveBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.candidates = []

    def num_candidates(self):
        return len(self.candidates)

    def add_new(self, candidate):
        flag = 1
        action_idx = None
        for idx, old_candidate in enumerate(self.candidates):
            if candidate.action == old_candidate.action and candidate.error < old_candidate.error:  # 如果判断出来和之前的action一样的话，就不去做
                flag = 1
                action_idx = idx
                break
            elif candidate.action == old_candidate.action:
                flag = 0

        if flag == 1:
            if action_idx is not None:
                print(action_idx)
                self.candidates.pop(action_idx)
            self.candidates.append(candidate)
            self.candidates = sorted(self.candidates, key=lambda x: x.error)  # from small to large

        if len(self.candidates) > self.max_size:
            self.candidates.pop(-1)  # remove the last one

if args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth1':
    def basic_tree():
        tree = BinaryTree('', False)
        tree.insertLeft('', True)
        tree.insertRight('', True)

        #tree = BinaryTree('', True)

        return tree

elif args.tree == 'depth2_rml':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', True)

        return tree

elif args.tree == 'depth2_rmu':
    print('**************************rmu**************************')
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', False)
        tree.rightChild.insertLeft('', True)
        tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth2_rmu2':
    print('**************************rmu2**************************')
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        # tree.rightChild.insertLeft('', True)
        # tree.rightChild.insertRight('', True)

        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth2_sub':
    print('**************************sub**************************')
    def basic_tree():
        tree = BinaryTree('', True)

        tree.insertLeft('', False)
        tree.leftChild.insertLeft('', True)
        tree.leftChild.insertRight('', True)

        # tree.rightChild.insertLeft('', True)
        # tree.rightChild.insertRight('', True)

        return tree

# structure = []
#
# def inorder_structure(tree):
#     global structure
#     if tree:
#         inorder_structure(tree.leftChild)
#         structure.append(tree.is_unary)
#         inorder_structure(tree.rightChild)
# inorder_structure(basic_tree())
# print('tree structure', structure)
#
# structure_choice = []
# for is_unary in structure:
#     if is_unary == True:
#         structure_choice.append(len(unary))
#     else:
#         structure_choice.append(len(binary))
# print('tree structure choices', structure_choice)

if args.tree == 'depth1':
    def basic_tree():
        # tree = BinaryTree('', False)
        # tree.insertLeft('',True)
        # tree.insertRight('',True)

        tree = BinaryTree('', False)
        tree.insertLeft('',False)
        tree.insertRight('',False)
        tree.leftChild.insertLeft('',True)
        tree.leftChild.insertRight('',True)
        # tree.leftChild.leftChild.insertLeft('',True)
        # tree.leftChild.leftChild.insertRight('',True)
        # tree.leftChild.rightChild.insertLeft('',True)
        # tree.leftChild.rightChild.insertRight('',True)

        # tree.rightChild.insertLeft('',False)
        # tree.rightChild.insertRight('',False)
        # tree.rightChild.leftChild.insertLeft('',True)
        # tree.rightChild.leftChild.insertRight('',True)
        # tree.rightChild.rightChild.insertLeft('',True)
        # tree.rightChild.rightChild.insertRight('',True)
        tree.rightChild.insertLeft('',True)
        tree.rightChild.insertRight('',True)

        # tree = BinaryTree('', False)
        # tree.insertLeft('', False)
        # tree.insertRight('', True)
        # tree.leftChild.insertRight('',False)
        # tree.leftChild.insertLeft('',False)
        # tree.leftChild.leftChild.insertLeft('',True)
        # tree.leftChild.leftChild.insertRight('',True)
        # tree.leftChild.rightChild.insertLeft('',True)
        # tree.leftChild.rightChild.insertRight('',True)



        return tree

elif args.tree == 'depth2':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.insertRight('', True)
        return tree

elif args.tree == 'depth3':
    def basic_tree():
        tree = BinaryTree('', False)

        tree.insertLeft('', True)
        tree.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.insertLeft('', False)
        tree.leftChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.leftChild.leftChild.insertRight('', True)
        tree.leftChild.leftChild.rightChild.insertLeft('', False)
        tree.leftChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.leftChild.leftChild.rightChild.leftChild.insertRight('', True)

        tree.insertRight('', True)
        tree.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.insertLeft('', False)
        tree.rightChild.leftChild.leftChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.leftChild.leftChild.insertRight('', True)

        tree.rightChild.leftChild.insertRight('', True)
        tree.rightChild.leftChild.rightChild.insertLeft('', False)
        tree.rightChild.leftChild.rightChild.leftChild.insertLeft('', True)
        tree.rightChild.leftChild.rightChild.leftChild.insertRight('', True)
        return tree

structure = []
leaves_index = []
leaves = 0
count = 0

def inorder_structure(tree):
    global structure, leaves, count, leaves_index
    if tree:
        inorder_structure(tree.leftChild)
        structure.append(tree.is_unary)
        if tree.leftChild is None and tree.rightChild is None:
            leaves = leaves + 1
            leaves_index.append(count)
        count = count + 1
        inorder_structure(tree.rightChild)


inorder_structure(basic_tree())

print('leaves index:', leaves_index)

print('tree structure:', structure, 'leaves num:', leaves)

structure_choice = []
for is_unary in structure:
    if is_unary == True:
        structure_choice.append(len(unary))
    else:
        structure_choice.append(len(binary))
print('tree structure choices', structure_choice)

def reset_params(tree_params):
    # tree_params[0].data = torch.tensor([[-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
    # tree_params[1].data = torch.tensor([0.0])
    # tree_params[2].data = torch.tensor([[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
    # tree_params[3].data = torch.tensor([0.0])
    for v in tree_params:
        # v.data.fill_(10)
        # print("what is v",v.data)
        # print('v.data.shape0',v.data.flatten().size(dim=0))
        # if v.data.flatten().size(dim=0) <= 1:
        #     v.data.fill_(0.0)
        # else:
        #     v.data = torch.tensor([[0,0,0,0.1,0,0.1,0,0,0,0,0,0]])
        # print("what is v",v.data)
        v.data.normal_(0.0, 0.1)

def inorder(tree, actions):
    global count
    if tree:
        inorder(tree.leftChild, actions)
        action = actions[count].item()
        if tree.is_unary:
            action = action
            tree.key = unary[action]
            # print(count, action, func.unary_functions_str[action])
        else:
            action = action
            tree.key = binary[action]
            # print(count, action, func.binary_functions_str[action])
        count = count + 1
        inorder(tree.rightChild, actions)

def inorder_visualize(tree, actions, trainable_tree):
    global count, leaves_cnt
    if tree:
        leftfun = inorder_visualize(tree.leftChild, actions, trainable_tree)
        action = actions[count].item()
        # print('123', tree.key)
        if tree.is_unary:# and not tree.key.is_leave:
            if count not in leaves_index:
                midfun = unary_functions_str[action]
                # a = trainable_tree.learnable_operator_set[count][action].a.item()
                # b = trainable_tree.learnable_operator_set[count][action].b.item()
            else:
                midfun = unary_functions_str_leaf[action]
        else:
            midfun = binary_functions_str[action]

        count = count + 1
        rightfun = inorder_visualize(tree.rightChild, actions, trainable_tree)
        if leftfun is None and rightfun is None:
            w = []
            for i in range(dim):
                w.append(trainable_tree.linear[leaves_cnt].weight[0][i].item())
                #bias = trainable_tree.linear[leaves_cnt].bias[0].item()
            leaves_cnt = leaves_cnt + 1
            ## -------------------------------------- input variable element wise  ----------------------------
            expression = ''
            for i in range(0, dim):
                # print(midfun)
                x_expression = midfun.format('x'+str(i))
                expression = expression + ('{:.4f}*{}'+'+').format(w[i], x_expression)
            #expression = expression+'{:.4f}'.format(bias)
            expression = '('+expression+')'
            # print('visualize', count, leaves_cnt, action)
            return expression
        # elif leftfun is not None and rightfun is None:
        #     if '(0)' in midfun or '(1)' in midfun:
        #         return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
        #     else:
        #         return midfun.format('{:.4f}'.format(a), leftfun, '{:.4f}'.format(b))
        # elif tree.leftChild is None and tree.rightChild is not None:
        #     if '(0)' in midfun or '(1)' in midfun:
        #         return midfun.format('{:.4f}'.format(a), '{:.4f}'.format(b))
        #     else:
        #         return midfun.format('{:.4f}'.format(a), rightfun, '{:.4f}'.format(b))
        else:
            return midfun.format(leftfun, rightfun)
    else:
        return None

def get_function(actions):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder(computation_tree, actions)
    count = 0
    return computation_tree

def inorder_params(tree, actions, unary_choices):
    #print('actions:',actions)
    global count
    if tree:
        inorder_params(tree.leftChild, actions, unary_choices)
        action = actions[count].item()
        #print(tree.is_unary)
        if tree.is_unary:
            action = action
            tree.key = unary_choices[count][action]
        else:
            action = action
            # print(count)
            # print(action)
            # print(len(unary_choices[count]),len(unary)+action)
            tree.key = unary_choices[count][len(unary)+action]
        count = count + 1
        inorder_params(tree.rightChild, actions, unary_choices)

def get_function_trainable_params(actions, unary_choices):
    global count
    count = 0
    computation_tree = basic_tree()
    inorder_params(computation_tree, actions, unary_choices)
    count = 0
    return computation_tree

class unary_operation(nn.Module):
    def __init__(self, operator, is_leave):
        super(unary_operation, self).__init__()
        self.unary = operator
        #if not is_leave:
            # self.a = nn.Parameter(torch.Tensor(1))
            # self.a.data.fill_(1)
            # self.b = nn.Parameter(torch.Tensor(1))
            # self.b.data.fill_(0)
        self.is_leave = is_leave

    def forward(self, x):
        if self.is_leave:
            return self.unary(x)
        else:
            return self.unary(x)
            #return self.a*self.unary(x)+self.b

class binary_operation(nn.Module):
    def __init__(self, operator):
        super(binary_operation, self).__init__()
        self.binary = operator
    def forward(self, x, y):
        return self.binary(x, y)

leaves_cnt = 0

def compute_by_tree(tree, linear, x):
    if tree.leftChild == None and tree.rightChild == None: # leaf node
        global leaves_cnt
        transformation = linear[leaves_cnt]
        leaves_cnt = leaves_cnt + 1
        # print('after key:',tree.key(x))
        # print('weight:',transformation.weight.data)
        # print('bias:',transformation.bias.data)
        # print('after transform:',transformation(tree.key(x)))
        return transformation(tree.key(x))
    elif tree.leftChild is None and tree.rightChild is not None:
        return tree.key(compute_by_tree(tree.rightChild, linear, x))
    elif tree.leftChild is not None and tree.rightChild is None:
        return tree.key(compute_by_tree(tree.leftChild, linear, x))
    else:
        return tree.key(compute_by_tree(tree.leftChild, linear, x), compute_by_tree(tree.rightChild, linear, x))

class learnable_compuatation_tree(nn.Module):
    def __init__(self):
        super(learnable_compuatation_tree, self).__init__()
        self.learnable_operator_set = {}
        for i in range(len(structure)):
            self.learnable_operator_set[i] = []
            is_leave = i in leaves_index
            for j in range(len(unary)):
                self.learnable_operator_set[i].append(unary_operation(unary[j], is_leave))
            for j in range(len(binary)):
                self.learnable_operator_set[i].append(binary_operation(binary[j]))
        self.linear = []
        #print(self.learnable_operator_set)
        for num, i in enumerate(range(leaves)):
            linear_module = torch.nn.Linear(dim, 1, bias=False) #set only one variable
            #linear_module.weight.data.normal_(0, 1/math.sqrt(dim))
            linear_module.weight.data.fill_(0)
            #linear_module.bias.data.fill_(0)
            self.linear.append(linear_module)
        #print(self.linear)

    def forward(self, x, bs_action):
        # print(len(bs_action))
        global leaves_cnt
        leaves_cnt = 0
        # print('self.learnable_operator_set',self.learnable_operator_set)
        # print('self.linear',self.linear)
        # print('get_function_trainable_params(bs_action, self.learnable_operator_set)',get_function_trainable_params(bs_action, self.learnable_operator_set))
        function = lambda y: compute_by_tree(get_function_trainable_params(bs_action, self.learnable_operator_set), self.linear, y)
        out = function(x)
        leaves_cnt = 0
        return out

# trainable_tree = learnable_compuatation_tree()
# #trainable_tree = trainable_tree#.cuda()
#
# params = []
# for idx, v in enumerate(trainable_tree.learnable_operator_set):
#     if idx not in leaves_index:
#         for modules in trainable_tree.learnable_operator_set[v]:
#             for param in modules.parameters():
#                 params.append(param)
# for module in trainable_tree.linear:
#     for param in module.parameters():
#         params.append(param)
# reset_params(params)
# x = torch.tensor([[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0],[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0]])
# print('y:',trainable_tree(x, torch.tensor([0])))
# time.sleep(10)


class Controller(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.softmax_temperature = 5.0
        self.tanh_c = 2.5
        self.mode = True

        self.input_size = 20
        self.hidden_size = 50
        self.output_size = sum(structure_choice)

        self._fc_controller = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size,self.output_size))

    def forward(self,x):
        logits = self._fc_controller(x)

        logits /= self.softmax_temperature

        # exploration # ??
        if self.mode == 'train':
            logits = (self.tanh_c*F.tanh(logits))

        return logits

    def sample(self, batch_size=1, step=0):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """

        # [B, L, H]
        inputs = torch.zeros(batch_size, self.input_size)
        #print(inputs.shape)
        log_probs = []
        actions = []
        total_logits = self.forward(inputs)

        cumsum = np.cumsum([0]+structure_choice)
        for idx in range(len(structure_choice)):
            logits = total_logits[:, cumsum[idx]:cumsum[idx+1]]

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            #print(log_prob.shape)
            # print(probs)
            if step>=args.random_step:
                action = probs.multinomial(num_samples=1).data
            else:
                action = torch.randint(0, structure_choice[idx], size=(batch_size, 1))
            # print('old', action)
            if args.greedy is not 0:
                for k in range(args.bs):
                    if np.random.rand(1)<args.greedy:
                        choice = random.choices(range(structure_choice[idx]), k=1)
                        action[k] = choice[0]
            # print('new', action)
            #print(action.shape)
            selected_log_prob = log_prob.gather(
                1, tools.get_variable(action, requires_grad=False))
            #print(selected_log_prob.shape)
            log_probs.append(selected_log_prob[:, 0:1])
            actions.append(action[:, 0:1])

        log_probs = torch.cat(log_probs, dim=1)   # 3*18
        #print(actions)
        #print(log_probs)
        return actions, log_probs

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (tools.get_variable(zeros, True, requires_grad=False),
                tools.get_variable(zeros.clone(), True, requires_grad=False))

def get_reward(bs, actions, learnable_tree, tree_params, tree_optim, indices):

    #x = (torch.rand(args.domainbs, dim))*(args.right-args.left)+args.left
    x = X_rnn[indices]
    # obs = obs_rnn[indices]#.reshape(args.domainbs,1)
    y = y_rnn[indices]#.reshape(args.domainbs,1)
    # p = partial_rnn[indices]*0
    print('x.shape:',x.shape)
    # print('obs.shape:',obs.shape)
    print('y.shape:',y.shape)
    # print('X[:10]',x[:10])
    # print('y[:10]',y[:10])
    pred = -0.04 * x[:,0] + 1 * x[:,1] * x[:,2]
    # print('pred[:10]', pred[:10])
    # print('x:',x)
    print('pred shape:',pred.shape)
    # print('y:',y)
    loss = nn.MSELoss()
    val = torch.sqrt(loss(pred, y))  # Convert to RMSE
    val = torch.std(y) * val  # Normalize using stdev of targets
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
    #val = 1 / (1 + val)
    print('true loss',val.item())
    # obs = obs_rnn[indices].reshape(args.domainbs,1)
    y = y_rnn[indices].reshape(args.domainbs,1)
    # p = partial_rnn[indices].reshape(args.domainbs,1)*0
    #print(x.shape)
    x.requires_grad = True

    # print(x)
    regression_errors = []
    formulas = []
    batch_size = bs

    global count, leaves_cnt
    #print(actions)

    for bs_idx in range(batch_size):

        bs_action = [v[bs_idx] for v in actions]
        bs_action = [torch.tensor([0]),torch.tensor([1]),torch.tensor([0]),torch.tensor([0]),torch.tensor([0]),torch.tensor([0]),torch.tensor([0])]
        # bs_action = [torch.tensor([0]),torch.tensor([1]),torch.tensor([0]),torch.tensor([2]),torch.tensor([0]),torch.tensor([1]),torch.tensor([0]),torch.tensor([0]),torch.tensor([0])]
        # print('bs_action:',bs_action)


        reset_params(tree_params)
        tree_optim = torch.optim.Adam(tree_params, lr=1e-1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(tree_optim, step_size=1000, gamma=0.5)
        # print('tree param after reset:', tree_params)
        for _ in range(6000):
            #bd_pts = get_boundary(args.bdbs, dim)
            #print(bd_pts.shape)
            #bc_true = func.true_solution(bd_pts)
            #print(bs_action)
            #bd_nn = learnable_tree(bd_pts, bs_action)
            count = 0
            #bd_error = torch.nn.functional.mse_loss(bc_true, bd_nn)
            #function_error = torch.nn.functional.mse_loss(func.LHS_pde(learnable_tree(x, bs_action), x, dim), func.RHS_pde(x))
            #loss = function_error + 100*bd_error
            # print('output of learnable_tree:',learnable_tree(x,bs_action),learnable_tree(x,bs_action).shape)
            y_pred = learnable_tree(x,bs_action)
            # y_pred=28 * x[:, 0] - x[:, 0] * x[:, 2] - x[:, 1]
            # print('X[:10]', x[:10])
            # print('y[:10]', y[:10])
            # y_pred = x[:,0]*(28-x[:,2])-x[:,1]
            y_pred = y_pred.reshape(args.domainbs,1)
            # print('y_pred[:10]', y_pred[:10])
            # print('y_pred shape:',y_pred.shape)
            loss = nn.MSELoss()
            val = torch.sqrt(loss(y_pred, y))  # Convert to RMSE
            val = torch.std(y) * val  # Normalize using stdev of targets
            val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
            # print('true loss1', val.item())
            # time.sleep(1)
            #val = 1 / (1 + val)
            #loss = torch.nn.functional.mse_loss(learnable_tree(x,bs_action)*globalnames['dt']+obs,y)
            loss = val
            #print('loss:',loss.item())
            tree_optim.zero_grad()
            loss.backward()
            # time.sleep(1)
            tree_optim.step()
            # print('tree_params:',tree_params)
            lr_scheduler.step()
            #print('tree param:',tree_params)

        tree_optim = torch.optim.Adam(tree_params, lr=1e-4)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(tree_optim, step_size=10, gamma=0.8)
        print('---------------------------------- batch idx {} -------------------------------------'.format(bs_idx))

        error_hist = []
        def closure():
            tree_optim.zero_grad()
            #bd_pts = get_boundary(args.bdbs, dim)
            #bc_true = func.true_solution(bd_pts)
            #bd_nn = learnable_tree(bd_pts, bs_action)
            #bd_error = torch.nn.functional.mse_loss(bc_true, bd_nn)
            #function_error = torch.nn.functional.mse_loss(func.LHS_pde(learnable_tree(x, bs_action), x, dim), func.RHS_pde(x))
            #loss = function_error + 100*bd_error
            y_pred = learnable_tree(x,bs_action)
            loss = nn.MSELoss()
            val = torch.sqrt(loss(y_pred, y))  # Convert to RMSE
            val = torch.std(y_rnn) * val  # Normalize using stdev of targets
            val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
            #val = 1 / (1 + val)
            #loss = torch.nn.functional.mse_loss(learnable_tree(x,bs_action)*globalnames['dt']+obs,y)
            loss = val
            print('loss before: ', loss.item())
            error_hist.append(loss.item())
            loss.backward()
            return loss

        tree_optim.step(closure)


        #function_error = torch.nn.functional.mse_loss(func.LHS_pde(learnable_tree(x, bs_action), x, dim), func.RHS_pde(x))
        #bd_pts = get_boundary(args.bdbs, dim)
        #bc_true = func.true_solution(bd_pts)
        #bd_nn = learnable_tree(bd_pts, bs_action)
        #bd_error = torch.nn.functional.mse_loss(bc_true, bd_nn)
        #regression_error = function_error + 100*bd_error
        y_pred = learnable_tree(x, bs_action)
        loss = nn.MSELoss()
        val = torch.sqrt(loss(y_pred, y))  # Convert to RMSE
        val = torch.std(y_rnn) * val  # Normalize using stdev of targets
        val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
        #val = 1 / (1 + val)
        # loss = torch.nn.functional.mse_loss(learnable_tree(x,bs_action)*globalnames['dt']+obs,y)
        regression_error = val
        #print('loss after, bd error: {}  '.format(bd_error.item()), ' eigen: {} '.format(function_error.item()))
        print('loss after, regression error: {}  '.format(regression_error.item()))
        error_hist.append(regression_error.item())

        print(error_hist, ' min: ', min(error_hist))
        regression_errors.append(min(error_hist))
        count = 0
        leaves_cnt = 0
        formula = inorder_visualize(basic_tree(), bs_action, trainable_tree)
        count = 0
        leaves_cnt = 0
        formulas.append(formula)
        #time.sleep(10)

    return regression_errors, formulas

def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

def true(x):
    return -0.5*(torch.sum(x**2, dim=1, keepdim=True))

def best_error(best_action, learnable_tree, indeces):

    #x = (torch.rand(args.domainbs, dim))*(args.right-args.left)+args.left
    x = X_rnn[indeces]
    obs = obs_rnn[indeces]
    y = y_rnn[indeces]
    x.requires_grad = True
    pred = ((x[:,3]+x[:,5])*globalnames['viscosity']-x[:,0]*x[:,1]-x[:,6]*x[:,2])*globalnames['dt']+obs
    loss = nn.MSELoss()
    val = torch.sqrt(loss(pred, y))  # Convert to RMSE
    val = torch.std(y_rnn) * val  # Normalize using stdev of targets
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
    #val = 1 / (1 + val)
    print('true error',val.item())

    obs = obs_rnn[indeces].reshape(args.domainbs,1)
    y = y_rnn[indeces].reshape(args.domainbs,1)

    bs_action = best_action

    #bd_pts = get_boundary(args.bdbs, dim)
    #bc_true = func.true_solution(bd_pts)
    #bd_nn = learnable_tree(bd_pts, bs_action)
    #bd_error = torch.nn.functional.mse_loss(bc_true, bd_nn)
    #function_error = torch.nn.functional.mse_loss(func.LHS_pde(learnable_tree(x, bs_action), x, dim), func.RHS_pde(x))
    #regression_error = function_error + 100 * bd_error #func.LHS_pde(true(x), x, dim)#func.LHS_pde(learnable_tree(x, bs_action), x, dim)# func.LHS_pde(true(x), x, dim)#func.LHS_pde(learnable_tree(x, bs_action), x, dim)
    y_pred = learnable_tree(x, bs_action) * globalnames['dt'] + obs
    loss = nn.MSELoss()
    val = torch.sqrt(loss(y_pred, y))  # Convert to RMSE
    val = torch.std(y_rnn) * val  # Normalize using stdev of targets
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
    #val = 1 / (1 + val)
    # loss = torch.nn.functional.mse_loss(learnable_tree(x,bs_action)*globalnames['dt']+obs,y)
    regression_error = val
    #print('bd error: {}  '.format(bd_error.item()), ' eigen: {} '.format(function_error.item()))
    print('regression error: {}  '.format(regression_error.item()))
    return regression_error

def train_controller(controller, controller_optim,controller_lr_scheduler, trainable_tree, tree_params, hyperparams):

    ### obtain a new file name ###
    file_name = os.path.join(hyperparams['checkpoint'], 'log{}.txt')
    file_idx = 0
    while os.path.isfile(file_name.format(file_idx)):
        file_idx += 1
    file_name = file_name.format(file_idx)
    logger = Logger(file_name, title='')
    logger.set_names(['iteration', 'loss', 'baseline', 'error', 'formula', 'error'])

    model = controller
    model.train()

    baseline = None

    bs = args.bs
    smallest_error = float('inf')

    candidates = SaveBuffer(10)

    train_sampler = BatchSampler(len(X_rnn), shuffle=True)

    tree_optim = None#torch.optim.Adam(tree_params, lr=1)#torch.optim.LBFGS(tree_params)#torch.optim.Adam(tree_params, lr=0.001)#torch.optim.LBFGS([])

    for step in range(hyperparams['controller_max_step']):
        # sample models
        actions, log_probs = controller.sample(batch_size=bs, step=step)
        indices = train_sampler.get_next(args.domainbs)
        #actions = [torch.tensor([a]).repeat(10,1) for a in [0,1,0,2,0,1,0,0]]
        # actions = [torch.tensor([a]).repeat(10, 1) for a in [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]

        # get the action code (binary to decimal)
        binary_code = ''
        for action in actions:
            binary_code = binary_code + str(action[0].item())
            # print(binary_code)
        # print(actions, '**********************************************')
        rewards, formulas = get_reward(bs, actions, trainable_tree, tree_params, tree_optim, indices)
        rewards = torch.FloatTensor(rewards).view(-1,1)

        # discount
        if 1 > hyperparams['discount'] > 0:
            rewards = discount(rewards, hyperparams['discount'])

        base = args.base
        rewards[rewards > base] = base
        rewards[rewards != rewards] = 1e10
        error = rewards
        rewards = 1 / (1 + torch.sqrt(rewards))

        batch_smallest = error.min()
        batch_min_idx = torch.argmin(error)
        batch_min_action = [v[batch_min_idx] for v in actions]

        batch_best_formula = formulas[batch_min_idx]

        candidates.add_new(candidate(action=batch_min_action, expression=batch_best_formula, error=batch_smallest))

        for candidate_ in candidates.candidates:
            print('error:{} action:{} formula:{}'.format(candidate_.error.item(), [v.item() for v in candidate_.action], candidate_.expression))

        # moving average baseline
        if baseline is None:
            baseline = (rewards).mean()
        else:
            decay = hyperparams['ema_baseline_decay']
            baseline = decay * baseline + (1 - decay) * (rewards).mean()

        argsort = torch.argsort(rewards.squeeze(1), descending=True)
        # print(error, argsort)
        # print(rewards.size(), rewards.squeeze(1), torch.argsort(rewards.squeeze(1)), rewards[argsort])
        # policy loss
        num = int(args.bs * args.percentile)
        rewards_sort = rewards[argsort]
        adv = rewards_sort - rewards_sort[num:num + 1, 0:]  # - baseline
        # print(error, argsort, rewards_sort, adv)
        log_probs_sort = log_probs[argsort]
        # print('adv', adv)
        loss = -log_probs_sort[:num] * tools.get_variable(adv[:num], True, requires_grad=False)
        loss = (loss.sum(1)).mean()

        # update
        controller_optim.zero_grad()
        loss.backward()
        print('controller loss:',loss.item())

        if hyperparams['controller_grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          hyperparams['controller_grad_clip'])
        controller_optim.step()
        controller_lr_scheduler.step()

        min_error = error.min().item()
        if smallest_error>min_error:
            smallest_error = min_error

            min_idx = torch.argmin(error)
            min_action = [v[min_idx] for v in actions]
            best_formula = formulas[min_idx]


        log = 'Step: {step}| Loss: {loss:.4f}| Action: {act} |Baseline: {base:.4f}| ' \
              'Reward {re:.4f} | {error:.8f} {formula}'.format(loss=loss.item(), base=baseline, act=min_action,
                                                               re=(rewards).mean(), step=step, formula=best_formula,
                                                               error=smallest_error)
        print('********************************************************************************************************')
        print(log)
        print('********************************************************************************************************')
        if (step + 1) % 1 == 0:
            logger.append([step + 1, loss.item(), baseline, rewards.mean(), smallest_error, best_formula])

    for candidate_ in candidates.candidates:
        print('error:{} action:{} formula:{}'.format(candidate_.error.item(), [v.item() for v in candidate_.action],
                                                     candidate_.expression))
        logger.append([666, 0, 0, 0, candidate_.error.item(), candidate_.expression])

    finetune = 2000
    global count, leaves_cnt
    for candidate_ in candidates.candidates:
        trainable_tree = learnable_compuatation_tree()
        trainable_tree = trainable_tree#.cuda()

        params = []
        for idx, v in enumerate(trainable_tree.learnable_operator_set):
            if idx not in leaves_index:
                for modules in trainable_tree.learnable_operator_set[v]:
                    for param in modules.parameters():
                        params.append(param)
        for module in trainable_tree.linear:
            for param in module.parameters():
                params.append(param)
        reset_params(params)
        finetune_sampler = BatchSampler(len(X_rnn), shuffle=True)
        tree_optim = torch.optim.Adam(params, lr=1e-1, )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(tree_optim, step_size=40, gamma=0.8)
        #print(params)
        indices = finetune_sampler.get_next(args.domainbs)
        error = best_error(candidate_.action, trainable_tree, indices)
        count = 0
        leaves_cnt = 0
        formula = inorder_visualize(basic_tree(), candidate_.action, trainable_tree)
        leaves_cnt = 0
        count = 0
        suffix = 'Finetune-- Iter {current_iter} Error {error:.5f} Formula {formula}'.format(current_iter=-1,
                                                                                             error=error,
                                                                                             formula=formula)
        print(suffix)
        for current_iter in range(finetune):
            indices = finetune_sampler.get_next(args.domainbs)
            error = best_error(candidate_.action, trainable_tree,indices)
            tree_optim.zero_grad()
            error.backward()

            tree_optim.step()
            lr_scheduler.step()
            #print(params)
            count = 0
            leaves_cnt = 0
            formula = inorder_visualize(basic_tree(), candidate_.action, trainable_tree)
            leaves_cnt = 0
            count = 0
            suffix = 'Finetune-- Iter {current_iter} Error {error:.5f} Formula {formula}'.format(current_iter=current_iter, error=error, formula=formula)
            if (current_iter + 1) % 100 == 0:
                logger.append([current_iter, 0, 0, 0, error.item(), formula])

            #cosine_lr(tree_optim, 1e-3, current_iter, finetune)
            print(suffix)

        #numerators = []
        #denominators = []

        # for i in range(1000):
        #     print(i)
        #     x = (torch.rand(100000, args.dim)) * (args.right - args.left) + args.left
        #     sq_de = torch.mean((func.true_solution(x))**2)
        #     sq_nu = torch.mean((func.true_solution(x)-trainable_tree(x, candidate_.action)) ** 2)
        #     numerators.append(sq_nu.item())
        #     denominators.append(sq_de.item())
        numerators = torch.mean((y_rnn.reshape(-1,1)-trainable_tree(X_rnn, candidate_.action)*globalnames["dt"]+obs_rnn.reshape(-1,1)) ** 2)
        denominators = torch.mean((y_rnn.reshape(-1,1)) ** 2)
        relative_l2 = math.sqrt(numerators) / math.sqrt(denominators)
        print('relative l2 error: ', relative_l2)
        logger.append(['relative_l2', 0, 0, 0, relative_l2, 0])

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    controller = Controller()#. ()
    hyperparams = {}

    hyperparams['controller_max_step'] = args.epoch
    hyperparams['discount'] = 1.0
    hyperparams['ema_baseline_decay'] = 0.95
    hyperparams['controller_lr'] = args.lr
    hyperparams['entropy_mode'] = 'reward'
    hyperparams['controller_grad_clip'] = 0#10
    hyperparams['checkpoint'] = args.ckpt
    if not os.path.isdir(hyperparams['checkpoint']):
        mkdir_p(hyperparams['checkpoint'])
    controller_optim = torch.optim.Adam(controller.parameters(), lr= hyperparams['controller_lr'])
    controller_lr_scheduler = torch.optim.lr_scheduler.StepLR(controller_optim, step_size=40, gamma=0.9)
    trainable_tree = learnable_compuatation_tree()
    trainable_tree = trainable_tree#.cuda()

    params = []
    for idx, v in enumerate(trainable_tree.learnable_operator_set):
        if idx not in leaves_index:
            for modules in trainable_tree.learnable_operator_set[v]:
                for param in modules.parameters():
                    params.append(param)
    for module in trainable_tree.linear:
        for param in module.parameters():
            params.append(param)

    train_controller(controller, controller_optim,controller_lr_scheduler, trainable_tree, params, hyperparams)