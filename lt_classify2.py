import torch
import matplotlib.pyplot as plt
import examples.mnist as M
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse

from MCMC import makeUpdateMCMC


## ******************************************************
## ******************************************************
parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str, default='./model.p')
args = parser.parse_args()
 
model=torch.load(args.modelpath, map_location='cpu')
#model=torch.load("./model.p", map_location='cpu') #LT
print("Loaded model.p")


# ===== UPDATE WITH LABELS
k = open('./examples/mnist/numbers.txt', 'r')
a = k.readlines()
a = str(int(a[0]))
a = [int(aa) for aa in a]
M.datalabels = torch.tensor(a)

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



## ===== COMPARE original binarized dataset (above, train) vs. one that I made (below, test)

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
##    plt.tight_layout()

# Classification code below

## ======= define a function that loads appropriate characters
def getIdx(charSamp, M, Nway=20, testOrTrain='Test'):    
#     charSamp = 1 # the sample integer (0,...,9)
#     Nway = 20 # how many test characters (only 1 will match the Samp)

    charSamp = torch.tensor(np.array(charSamp))
    
    if testOrTrain=='Test':
        labthis = M.testlabels
    elif testOrTrain=='Train':
        labthis = M.datalabels
    
    # get sample
    a = np.where(labthis==charSamp)
    idx_sample, idx_testmatch = np.random.choice(a[0], size=2, replace=False)

    # get the N-1 test that do not match the sample
    a = np.where(labthis!=charSamp)
    idx_testnonmatch = np.random.choice(a[0], size=Nway-1, replace=False)
    
    return idx_sample, idx_testmatch, idx_testnonmatch




## ==================================================================
## ===== ZEROTH, extract indices in test set that you want to work with
def getClassScore(charSamp, Nway=10, frontierSize=5, nUpdates=10, plotON=False, 
                  noise=0, nIterNoise=0, testOrTrain='Test', redoUpdate=True, 
                  doMCMC=False, MCMC_strokesToFlip=range(1,7)):
# charSamp = 3 # which digit to use for the sample?
# Nway=15
# frontierSize = 5; how many components to remember

    if noise==0:
        # then no point doing more than one iteration.
        nIterNoise=1
        
    # ========== GET RANDOM SAMPLES
    # IN ORDER: (sample, same_as_sample, diff_from_sample)
    idx_sample, idx_testmatch, idx_testnonmatch = getIdx(charSamp = charSamp, M=M, 
                                                         Nway=Nway, testOrTrain=testOrTrain)
    idx_all = np.concatenate((idx_sample.reshape(1), idx_testmatch.reshape(1), idx_testnonmatch))

    if testOrTrain=='Test':
        datthis = M.testdata
    elif testOrTrain=='Train':
        datthis = M.data

    # EXTRACT DATA
    x_samp = datthis[idx_all[0]] # sample
#     x_test = [M.testdata[ii] for ii in idx_all[1:]] # need to already be updated in the model.
    x_all = [datthis[ii] for ii in idx_all] # need to already be updated in the model.
    # idx_all = range(len(idx_all))


    # ====== FIRST, EMPTY MIXTURE COMPONENTS
    if redoUpdate:
        model.frontierSize = frontierSize
        from torch.nn import Parameter
        model.mixtureComponents = [[] for _ in range(len(datthis))]
        model.mixtureWeights = Parameter(torch.zeros(len(datthis), model.frontierSize)) #Unnormalised log-q
        model.mixtureScores = [[] for _ in range(len(datthis))] #most recent log joint
        model.nMixtureComponents = Parameter(torch.zeros(len(datthis)))
    
    
        ## ===== FIRST, need to update model with posteriors for the novel stimuli    
        if plotON is True:
            print(model.mixtureWeights[idx_all[0]])
        model.makeUpdates(i=idx_all, x=x_all, nUpdates=nUpdates)
        if plotON is True:
            print(model.mixtureWeights[idx_all[0]])
        
        ## ===== DO MCMC?
#        if doMCMC is True:
 #           for jj in MCMC_strokesToFlip:
  #              print('doing mcmc(flipping) for strokenum %s' %jj)
   #             model = makeUpdateMCMC(model, i=idx_all, x=x_all, strokeNum=jj)
            
    else:
        # == then this must be on Training set for the indices to be correct
        assert(testOrTrain=='Train')

    ## TESTING CLASSIFICATION ACCURACY
    
    # x_samp = M.testdata[idx_all[0]] # sample
    # x_test = [M.testdata[ii] for ii in idx_all] # need to already be updated in the model.
    # scores = model.conditional(idx_all, x_test, [x_samp]*(len(idx_all)-1))
    scores_all = []
    scores_sampmodel_all = []
    for _ in range(nIterNoise):
        scores = model.conditional(i=idx_all, D=x_all, x=[x_samp]*(len(idx_all)), 
                                   noise=noise) # use model for D to predict x_samp
        
        scores_sampmodel = [model.conditional(i=np.array(idx_all[0:1]), D=x_all[0:1], 
                                              x=x_all[kk:kk+1], noise = noise) for kk in range(0, len(x_all))]
        scores_sampmodel = torch.tensor(np.concatenate([aa.detach().numpy() for aa in scores_sampmodel]))

        scores_all.append(scores)
        scores_sampmodel_all.append(scores_sampmodel)
        
    
    # ==== for scores and scores_sampmodel, find maximum for each character
    scores_all = np.array([ss.detach().numpy() for ss in scores_all])
    scores = scores_all.max(0)
    
    scores_sampmodel_all = np.array([ss.detach().numpy() for ss in 
                                     scores_sampmodel_all])
    scores_sampmodel = scores_sampmodel_all.max(0)
        
    scores_sum = scores + scores_sampmodel
    # ======== FOR EACH TEST CHARACTER, GET ITS SCORE 
    if plotON is True:
        plt.figure()
        plt.plot(scores, '-ok')
        # predictive += scores[0].mean().item() #.item() so we don't hold onto tensor for gradient information
        # if not (scores>scores[0]).any():
        #     num_tied = (scores==scores[0]).sum().item()
        #     hits += 1/num_tied 
        # total += 1

        # print("Took", int(time.time()-starttime), "seconds")
        # print(hits / total) # hit rate
        # print(predictive/total) # mean score
    print(scores)
    print(idx_all)
    return scores, scores_sampmodel, idx_all, scores_sum





## =====================================================
Ntrials = 15
import pickle
charall = np.random.randint(0, 10, Ntrials)
noise = 0.25
nIterNoise = 5 # how many times to sample, only has effect if noise>0
saveInterval = 20 # save mod this number trials
testOrTrain = 'Train'
redoUpdate = True
doMCMC = False
MCMC_strokesToFlip = range(1,7) # which strokes to flip [ignore first, since it is always off]

# for w, f, u in zip([5, 5, 10, 10, 20, 20],[5, 10, 5, 10, 5, 10],[20, 20, 20, 20, 20, 20]):
for noise in [0.25]:
    for testOrTrain in ['Train']:
        for w, f, u in zip([20],[10],[20]):
            scores_all = []
            scores_sampmodel_all = []
            scores_sum_all = []
            idx_all = []
            
            if noise==0:
                fname = './datsave/datall_w%sf%su%s' %(w, f, u) + testOrTrain
            elif noise>0:
                fname = './datsave/datall_w%sf%su%sn%snlev%s' %(w, f, u, nIterNoise, noise) + testOrTrain
            
            if not redoUpdate:
                fname = fname + 'noRedoUpdate'
                
            if doMCMC is True:
                fname = fname + 'MCMC%s' %MCMC_strokesToFlip[-1]
            
            print(fname)
            
            
            
            for i, cc in enumerate(charall):
                print(i)
                cc = np.array(cc)
                scores, scores_sampmodel, idx, scores_sum = \
                getClassScore(charSamp=cc, Nway=w, frontierSize=f, nUpdates=u, plotON=False, 
                              noise = noise, nIterNoise=nIterNoise, testOrTrain=testOrTrain, 
                              redoUpdate=redoUpdate, doMCMC=doMCMC)
                              
        
                scores_all.append(scores)
                scores_sampmodel_all.append(scores_sampmodel)
                scores_sum_all.append(scores_sum)
                idx_all.append(idx)
        
                if i%saveInterval==0 and i>0:
                    ## ========= SAVE OUTPUT
                    datall = [scores_all, scores_sampmodel, scores_sampmodel_all, idx_all]
        
                    # ======== SAVE
                    with open(fname , 'wb') as fl:
                        pickle.dump(datall, fl)
        
                    print('==== SAVED!')
                        
                        
                ## ========= SAVE FINAL
                datall = [scores_all, scores_sampmodel, scores_sampmodel_all, idx_all]
                with open(fname, 'wb') as fl:
                    pickle.dump(datall, fl)
    
