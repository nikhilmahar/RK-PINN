# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:22:17 2022

@author: General
"""

from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def save_gif_PIL(outfile, files, fps=5, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)
    
def exact_sol(t,w0):
    ty  = (-torch.cos(w0*t)+1)/w0
    return ty
def sin_t(t,w0):
    y  = torch.sin(w0*t)
    return y
N = nn.Sequential(nn.Linear(1, 20,bias=True),
                  nn.Tanh(), 
                  nn.Linear(20,20,bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,4, bias=True))
    
w0 =  torch.tensor([1])       #omega=1
it=0      #starting time
ft=100        #ending time
par=100   # total no. of partitions
# get the analytical solution over the full domain
tt = torch.linspace(it,ft,par)      # true time vector
ty = exact_sol(tt, w0)              #true solution vector
# print(t.shape, y.shape)

# ic_data=torch.zeros(1,1)



# t_physics = torch.linspace(0,1,10).view(-1,1).requires_grad_(True) # sample locations over the problem domain
t0=torch.zeros(1,1)     #initial time value
y0=torch.zeros(1,1)     #initial solution value
y=torch.zeros((1,1))
# t=torch.linspace(0, 2,5).reshape(5,1)
t=tt.reshape(len(tt),1)     #time vector for pinn
# yv=torch.zeros((4,1))
yv=torch.zeros((len(tt)-1,1))   #vector to store solution y after each time step
# t1,y1=torch.zeros(5,1),torch.zeros(5,1)
t[0,:],y[0,:]=t0,y0
# ty=torch.cat((t,y),axis=1).requires_grad_(True)
# print("ty_shape",ty.shape)
# delta_t=torch.tensor((max(t)-min(t))/len(t))
delta_t=torch.tensor(t[1]-t[0])    #time step value
# ty=torch.cat((t0,y0),axis=1)
torch.manual_seed(123)
# k_pred=N(ty).reshape(4,1)
optimizer = torch.optim.Adam(N.parameters(),lr=1e-4)
files = []
def rkcell(k_pred,t,y):
    # t0_rk=torch.zeros(1,1)
    # y0_rk=torch.zeros(1,1)
    # print("predicted k",k_pred)
    # print("called in rk t",t)
    # print("caled in rk y",y)
    A=torch.tensor([[0, 0, 0, 0],
       [0.5,0,0,0],
       [0,0.5,0,0],
       [0,0,1,0]])
    B=torch.tensor([[0],[0.5],[0.5],[1]])
    C=torch.tensor([1/6,1/3,1/3,1/6])
    # k_rk=delta_t*sin_t(t+B*delta_t,y+torch.matmul(A,k_pred))
    k_rk=delta_t*sin_t(t+B*delta_t,w0)
    y=torch.matmul(C,k_rk)
    # print("y",y)
    if i==0:
        yv[i,:]=y
    else:
        yv[i,:]=y+yv[i-1,:]
        y=y+yv[i-1,:]
    
    
    # print("yv",yv)
    # print("y+yv",y)
    # t=t+delta_t
    return k_rk,t,y,yv
epochs=50
# Let's see now if a stochastic optimizer makes a difference
adam = torch.optim.Adam(N.parameters(), lr=0.001)

for i in range(0,len(t)-1):
    # if i==0:
    
        # print("Time step",i+1)
        # print("t",t,"brfore rk y",y) 
        def loss(t,y):
            t=t
            y=y
            # t.requires_grad=True
            k_pred=N(y)
            k_pred=k_pred.reshape(4,1)
            # print("y",y)
            # print("k_pred",k_pred)
            k_rk,t1,y1,yv1=rkcell(k_pred,t,y)
            # t,y=t1,y1
            # print("y after rk",y1)
            loss=torch.mean((k_pred-k_rk)**2)
            if epochs/(epoch+1)==1:
                print("time step=",i+1,"loss",loss.item())
            return loss, t1,y1,yv1
           
        
        for epoch in range(epochs):
                # print("epoch",epoch+1)
                # tt=loss(ty,t,y)
                adam.zero_grad()
            #     # Evaluate the loss
                # print("ty",ty)
                # tyi=ty[i,:]
                # tyi=tyi.reshape(1,2)
                # print("tyi",tyi)
                
                l,t2,y2,yv2 = loss(t[i,:],y)
                # print(l)
                # print("after rk and loss y",y2)
                
            #     # Calculate the gradients
                l.backward(retain_graph=True)
                # with torch.no_grad():
                    # tyi[:,1]=y2                
            #         # Update the network
                adam.step()
              # print("t",t,"afetr epoch y",y)
        with torch.no_grad():
            y=y2
plt.figure()
# plt.plot(tt[1:4,:], ty[1:4,:], label="Exact solution")
# plt.plot(t[1:4,:], yv[0:3,:], label="PINN solution")  
plt.plot(tt[it+1:par-1], ty[it+1:par-1], label="Exact solution")
plt.plot(t[it+1:par-1,:], yv[it:par-2,:], label="PINN solution")  
plt.legend()
plt.show()
         
# y_test=torch.tensor([0.25])
# k_test=N(y_test)
# print("k_test",k_test)

    

