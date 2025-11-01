import sys,time
import numpy as np
import torch
import timeout_decorator
from aTEAM.optim import NumpyFunctionInterface
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
from scipy.optimize.slsqp import fmin_slsqp as slsqp
from scipy.optimize import fmin_bfgs as bfgs
import conf,setenv,initparameters
import random
import torch.nn as nn
from DSRtrain import train
import matplotlib.pyplot as plt
#%%
kw = None
kw = {
        '--name':'test1',
        '--dtype':'double',
        '--device':'cpu',
        '--constraint':2,
        # computing region
        '--eps':2*np.pi,
        '--dt':1e-2,
        '--cell_num':1,
        '--blocks':'0-6,9,12,15,18,20',
        # super parameters of network
        '--kernel_size':5,
        '--max_order':2,
        '--dx':2*np.pi/32,
        '--hidden_layers':3,
        '--scheme':'upwind',
        # data generator
        '--dataname':'burgers',
        '--viscosity':0.05,
        '--zoom':1,
        '--max_dt':1e-2,
        '--batch_size':1,
        '--data_timescheme':'euler',
        '--channel_names':'u,v',
        '--freq':4,
        '--data_start_time':1.0,
        # data transform
        '--start_noise':0.00,
        '--end_noise':0.00,
        # others
        '--stablize':0.0,
        '--sparsity':0.005,
        '--momentsparsity':0.001,
        '--npseed':-1,
        '--torchseed':-1,
        '--maxiter':2000,
        '--recordfile':'None',
        '--recordcycle':200,
        '--savecycle':-1,
        '--start_from':-1,
        }
options = conf.setoptions(argv=sys.argv[1:],kw=kw,configfile=None)

print(options)
globalnames, callback, model, data_model, sampling, addnoise = setenv.setenv(options)

globals().update(globalnames)

torch.cuda.manual_seed_all(torchseed)
torch.manual_seed(torchseed)
np.random.seed(npseed)
if start_from<0:
    initparameters.initkernels(model, scheme=scheme)
    # initparameters.renormalize(model, u0)
    # initparameters.initexpr(model, viscosity=viscosity, pattern='random')
else: # load checkpoint of layer-$start_from
    callback.load(start_from, iternum='final')

# print("convolution moment and kernels")
# for k in range(max_order+1):
#     for j in range(k+1):
#         print((model.__getattr__('fd'+str(j)+str(k-j)).moment).data.cpu().numpy())
#         print((model.__getattr__('fd'+str(j)+str(k-j)).kernel).data.cpu().numpy())

block = 25
u_obs,u_true,u = \
            setenv.data(model,data_model,globalnames,sampling,addnoise,block,data_start_time)
print("u_obs shape: batchsize x channelNum x xgridsize x ygridsize")
print(len(u_obs))
print(u_obs[0].shape)
print("u_obs.abs().max()")
print(u_obs[0].abs().max())
print("u_obs variance")
print(initparameters.trainvar(model.UInputs(u_obs[0])))
# print("u_obs")
# print(u_obs[0][:,0,:,:])
fd = model(u_obs[0],globalnames["dt"]).detach().numpy()
for i in range(1,len(u_obs)-1):
        fd_i = model(u_obs[i],globalnames["dt"]).detach().numpy()
        # print("fd_i shape:"),print(fd_i.shape)
        fd = np.concatenate((fd,fd_i),axis=0)
print('fd shape: batchsize x channelNum x xgridsize x ygridsize')
print(fd.shape)
X = fd.transpose(0,2,3,1).reshape(-1,12)
#print(X[28*32*32:28*32*32+32*32,0].reshape(32,32)-fd[28,0,:,:])
print("X.shape:")
print(X.shape)
y = u_obs[1][:,0,:,:].reshape(-1,1).detach().numpy()
for i in range(2,len(u_obs)):
        y_i = u_obs[i][:,0,:,:].reshape(-1,1).detach().numpy()
        y = np.concatenate((y,y_i),axis=0)
print("y.shape:")
print(y.shape)
#print(y[28*32*32:28*32*32+32*32,0].reshape(32,32)-u_obs[2][0,0,:,:].detach().numpy())
obs = u_obs[0][:,0,:,:].reshape(-1,1).detach().numpy()
for i in range(1,len(u_obs)-1):
        obs_i = u_obs[i][:,0,:,:].reshape(-1,1).detach().numpy()
        obs = np.concatenate((obs,obs_i),axis=0)
print("obs.shape:")
print(obs.shape)

#partial = -X[:,6]*X[:,2]-X[:,0]*X[:,1]+globalnames['viscosity']*(X[:,3]+X[:,5])
partial = (-X[:,6]*X[:,2]-X[:,0]*X[:,1]+globalnames['viscosity']*(X[:,3]+X[:,5]))
partial = partial.reshape(-1,1)*globalnames['dt']+obs
loss = nn.MSELoss()
val = torch.sqrt(loss(torch.tensor(partial), torch.tensor(y)))  # Convert to RMSE
val = torch.std(torch.tensor(y)) * val  # Normalize using stdev of targets
val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10))  # Fix nan and clip
# val = 1 / (1 + val)
print('true loss', val.item())
time.sleep(10)

start = 0
end = 1024
pred = obs[start:end].flatten()+(-X[start:end,0]*X[start:end,1]-X[start:end,6]*X[start:end,2]+globalnames['viscosity']*(X[start:end,3]+X[start:end,5]))*globalnames['dt']
print(pred.shape)
# pred_hat = obs[start:end].flatten()+((X[start:end,6]/1.3271)*((-0.0205*X[start:end,6])-X[start:end,2]))*1e-2
pred_hat = obs[start:end].flatten()+(-0.05*X[start:end,3]-X[start:end,0]*X[start:end,1]-X[start:end,2])*globalnames['dt']
print(pred_hat.shape)
input = obs[start:end].flatten()
true = y[start:end]
#
# plt.jet()
# plt.figure(figsize=(25,40))
#
# plt.subplot(4,2,1)
# plt.pcolor(true.reshape(32,32))
# plt.colorbar(fraction=0.05)
# plt.title("Truth",fontsize=20)
# plt.subplot(4,2,2)
# plt.pcolor(np.abs(true.reshape(32,32)-true.reshape(32,32)))
# plt.colorbar(fraction=0.05)
# plt.title("Truth error",fontsize=20)
#
# plt.subplot(4,2,3)
# plt.pcolor(input.reshape(32,32))
# plt.colorbar(fraction=0.05)
# plt.title("input",fontsize=20)
# plt.subplot(4,2,4)
# plt.pcolor(np.abs(input.reshape(32,32)-true.reshape(32,32)))
# plt.colorbar(fraction=0.05)
# plt.title("input error",fontsize=20)
#
# plt.subplot(4,2,5)
# plt.pcolor(pred_hat.reshape(32,32))
# plt.colorbar(fraction=0.05)
# plt.title("pred_hat",fontsize=20)
# plt.subplot(4,2,6)
# plt.pcolor(np.abs(pred_hat.reshape(32,32)-true.reshape(32,32)))
# plt.colorbar(fraction=0.05)
# plt.title("pred_hat error",fontsize=20)
#
# plt.subplot(4,2,7)
# plt.pcolor(pred.reshape(32,32))
# plt.colorbar(fraction=0.05)
# plt.title("Pred",fontsize=20)
# plt.subplot(4,2,8)
# plt.pcolor(np.abs(pred.reshape(32,32)-true.reshape(32,32)))
# plt.colorbar(fraction=0.05)
# plt.title("Point-wise Error",fontsize=20)
# plt.savefig("output.png")

# y = (y - obs)/globalnames["dt"]

# Split randomly
comb = list(zip(X, y, obs, partial))
random.shuffle(comb)
X, y ,obs,partial= zip(*comb)

# Proportion used to train constants versus benchmarking functions
training_proportion = 0.2

div = int(training_proportion * len(X))
X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
obs_constants, obs_rnn = np.array(obs[:div]), np.array(obs[div:])
partial_constants, partial_rnn = np.array(partial[:div]), np.array(partial[div:])
X_constants, X_rnn = torch.Tensor(X_constants), torch.Tensor(X_rnn)
y_constants, y_rnn = torch.Tensor(y_constants), torch.Tensor(y_rnn)
obs_constants, obs_rnn = torch.Tensor(obs_constants), torch.Tensor(obs_rnn)
partial_constants, partial_rnn = torch.Tensor(partial_constants), torch.Tensor(partial_rnn)
y_constants = torch.flatten(y_constants)
y_rnn = torch.flatten(y_rnn)
obs_constants = torch.flatten(obs_constants)
obs_rnn = torch.flatten(obs_rnn)
partial_constants = torch.flatten(partial_constants)
partial_rnn = torch.flatten(partial_rnn)
print('X_constants:shape')
print(X_constants.shape)
print('y_constants shape')
print(y_constants.shape)
print('obs_constants shape')
print(obs_constants.shape)
print("partial shape")
print((partial_constants.shape))

results = train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        obs_constants,
        obs_rnn,
        partial_constants,
        partial_rnn,
        operator_list=['*', '+','-','/','c','var_x00', 'var_x01', 'var_x10', 'var_x02', 'var_x11', 'var_x20', 'var_y00', 'var_y01', 'var_y10', 'var_y02', 'var_y11', 'var_y20'],
        min_length=3,
        max_length=15,
        type='lstm',
        num_layers=10,
        hidden_size=350,
        dropout=0.00,
        #lr=0.0005,
        lr = 1e-4,
        optimizer='adam',
        inner_optimizer='rmsprop',
        inner_lr=0.05,
        inner_num_epochs=50,
        entropy_coefficient=0.005,
        risk_factor=0.95,
        initial_batch_size=2000,
        scale_initial_risk=True,
        batch_size=1000,
        num_batches=1000,
        use_gpu=False,
        apply_constraint=False,
        live_print=True,
        summary_print=True

)

# Unpack results
epoch_best_rewards = results[0]
epoch_best_expressions = results[1]
best_reward = results[2]
best_expression = results[3]

# Plot best rewards each epoch
plt.plot([i + 1 for i in range(len(epoch_best_rewards))], epoch_best_rewards)
plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Reward over Time')
plt.show()
