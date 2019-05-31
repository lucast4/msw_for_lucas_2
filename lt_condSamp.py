#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:05:07 2019

Given new sample, does conditional generation. Can use either Test or Training set

@author: lucastian
"""


# 1) populate mixture components
# 2) generate sample

## ================ INPUT ARGUMNETS
frontierSize = 10
testOrTrain = 'Test' # which dataset
idxs = range(20,30) # which indices to use
nUpdates = 20
noise = 0.25
sample_probs = True
nSamps = 5

## ====
import matplotlib.pyplot as plt
import torch


## ================= LOAD TRAINED MODEL
model=torch.load("./model.p", map_location='cpu') #LT
print("Loaded model.p")


## ================== LOAD TRAIN AND TEST DATASETS
import examples.mnist as M

# ===== UPDATE WITH LABELS
k = open('./examples/mnist/numbers.txt', 'r')
a = k.readlines()
a = str(int(a[0]))
a = [int(aa) for aa in a]
M.datalabels = a

if False:
    # plot examples to confirm labels correct
    for j in range(5):
        plt.figure()
        for i, ii in enumerate(np.random.choice(len(M.data), 10)):
            plt.subplot(1,10, i+1)
            plt.imshow(M.data[ii][0])
            plt.title(M.datalabels[ii])

# ------ REPLACE M.testdata WITH TORCHVISION MNIST DATA, 
# so that have accurate labels.
# (scale by 255, sample from bernoulli with those p, in order to binarize.)
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 1) Load data
mnist_test = datasets.MNIST(root='./data', train=False, download=True)
mnist_test.test_data = mnist_test.test_data.reshape(-1, 1, 28, 28)

# 2) Binarize dataset
from torch.distributions.bernoulli import Bernoulli
mnist_bin = Bernoulli(mnist_test.test_data.to(torch.float64)/255).sample()
mnist_test.test_data = mnist_bin
mnist_test.test_data.to(torch.uint8)

# 3) Replace M with new labeled dataset
n = M.testdata.shape[0]
M.testdata = mnist_test.test_data[0:n]
M.testlabels = mnist_test.test_labels[0:n]


# 4) OPTIONAL: Plot to compare original binarized dataset (above, train) vs.
# one that I made (below, test)
if False:
    plt.figure(figsize=(10,20))
    indthis = np.random.randint(low=0, high=100, size=10)
    for i, ind in enumerate(indthis):
        plt.subplot(1,10,i+1)
        plt.imshow(M.data[ind][0].numpy().reshape(28, 28), vmin=0, vmax=1)
        plt.tight_layout()
        
    plt.figure(figsize=(10,20))
    for i, ind in enumerate(indthis):
        plt.subplot(1,10,i+1)
        plt.imshow(M.testdata[ind][0].numpy().reshape(28, 28), vmin=0, vmax=1)


## ================ CLEAR CURRENTLY SAVED MIXTURE COMPONENTS
model.frontierSize = frontierSize
from torch.nn import Parameter
model.mixtureComponents = [[] for _ in range(len(M.testdata))]
model.mixtureWeights = Parameter(torch.zeros(len(M.testdata), model.frontierSize)) #Unnormalised log-q
model.mixtureScores = [[] for _ in range(len(M.testdata))] #most recent log joint
model.nMixtureComponents = Parameter(torch.zeros(len(M.testdata)))

## ================ UPDATE MIXTURE COMPONENETS
# -- Get image data
if testOrTrain=='Train':
    x_all = [M.data[ii] for ii in idxs]
elif testOrTrain=='Test':
    x_all = [M.testdata[ii] for ii in idxs] # need to already be updated in the model.

model.makeUpdates(i=idxs, x=x_all, nUpdates=nUpdates)

print('sadfsdfsdfsaf')
## ================= GENERATE IMAGE
# ----- PLOT ORIGINAL IMAGES
# plt.figure(figsize=(10, 20))
plt.figure()
for i, xx in enumerate(x_all):
    plt.subplot(1,10,i+1)
    plt.title('c%s (gen)' % idxs[i])
    plt.imshow(xx.detach().numpy().reshape(28, 28), vmin=0, vmax=1)
plt.show()

# ----- PLOT GENERATED IMAGES
for j in range(nSamps):
    plt.figure()
    cs, xs = model.sample(i=idxs, x=x_all, noise=noise, sample_probs=sample_probs)
    for i, xx in enumerate(xs):
        plt.subplot(nSamps,10,i+1)
        plt.title('c%s (gen)' % idxs[i])
        plt.imshow(xx.detach().numpy().reshape(28, 28), vmin=0, vmax=1)
plt.show()

print('sdfsafdsafafdasfsfsadfasdfsdaf')