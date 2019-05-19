import torch
import matplotlib.pyplot as plt
import examples.mnist as M
import numpy as np

model=torch.load("./model.p", map_location='cpu') #LT
print("Loaded model.p")



## ==== REPLACE M.testdata WITH TORCHVISION MNIST DATA, so that have accurate labels.
# scale by 255, sample from bernoulli with those p, in order to binarize.


import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

mnist_test = datasets.MNIST(root='./data', train=False, download=True)
mnist_test.test_data = mnist_test.test_data.reshape(-1, 1, 28, 28)

# ===== BINARIZES THE DATASET
from torch.distributions.bernoulli import Bernoulli
mnist_bin = Bernoulli(mnist_test.test_data.to(torch.float64)/255).sample()

plt.figure()
plt.subplot(121)
plt.imshow(mnist_test.test_data.to(torch.float64)[1][0])
plt.subplot(122)
plt.imshow(mnist_bin[1][0])

mnist_test.test_data = mnist_bin

mnist_test.test_data.to(torch.uint8)
mnist_test.test_data[0][0]

plt.figure()
plt.subplot(121)
plt.imshow(mnist_test.test_data[1][0])

# ========= PERFORM REPLACEMENT
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

## define a function that loads appropriate characters

def getIdx(charSamp, M, Nway=20):    
#     charSamp = 1 # the sample integer (0,...,9)
#     Nway = 20 # how many test characters (only 1 will match the Samp)

    charSamp = torch.tensor(np.array(charSamp))
    
    # get sample
    a = np.where(M.testlabels==charSamp)
    idx_sample, idx_testmatch = np.random.choice(a[0], size=2, replace=False)

    # get the N-1 test that do not match the sample
    a = np.where(M.testlabels!=charSamp)
    idx_testnonmatch = np.random.choice(a[0], size=Nway-1, replace=False)
    
    return idx_sample, idx_testmatch, idx_testnonmatch




## ==================================================================
## ===== ZEROTH, extract indices in test set that you want to work with
def getClassScore(charSamp, Nway=10, frontierSize=5, nUpdates=10, plotON=False):
# charSamp = 3 # which digit to use for the sample?
# Nway=15
# frontierSize = 5; how many components to remember

    # ========== GET RANDOM SAMPLES
    # IN ORDER: (sample, same_as_sample, diff_from_sample)
    idx_sample, idx_testmatch, idx_testnonmatch = getIdx(charSamp = charSamp, M=M, Nway=Nway)
    idx_all = np.concatenate((idx_sample.reshape(1), idx_testmatch.reshape(1), idx_testnonmatch))


    # EXTRACT DATA
    x_samp = M.testdata[idx_all[0]] # sample
#     x_test = [M.testdata[ii] for ii in idx_all[1:]] # need to already be updated in the model.
    x_all = [M.testdata[ii] for ii in idx_all] # need to already be updated in the model.
    # idx_all = range(len(idx_all))


    # ====== FIRST, EMPTY MIXTURE COMPONENTS
    model.frontierSize = frontierSize
    from torch.nn import Parameter
    model.mixtureComponents = [[] for _ in range(len(M.testdata))]
    model.mixtureWeights = Parameter(torch.zeros(len(M.testdata), model.frontierSize)) #Unnormalised log-q
    model.mixtureScores = [[] for _ in range(len(M.testdata))] #most recent log joint
    model.nMixtureComponents = Parameter(torch.zeros(len(M.testdata)))


    ## ===== FIRST, need to update model with posteriors for the novel stimuli    
    if plotON is True:
        print(model.mixtureWeights[idx_all[0]])
    model.makeUpdates(i=idx_all, x=x_all, nUpdates=nUpdates)
    if plotON is True:
        print(model.mixtureWeights[idx_all[0]])

    ## TESTING CLASSIFICATION ACCURACY
    
    # x_samp = M.testdata[idx_all[0]] # sample
    # x_test = [M.testdata[ii] for ii in idx_all] # need to already be updated in the model.
    # scores = model.conditional(idx_all, x_test, [x_samp]*(len(idx_all)-1))
    scores = model.conditional(i=idx_all, D=x_all, x=[x_samp]*(len(idx_all))) # use model for D to predict x_samp
    
    scores_sampmodel = [model.conditional(i=np.array(idx_all[0:1]), D=x_all[0:1], x=x_all[kk:kk+1]) for kk in range(0, len(x_all))]
    scores_sampmodel = torch.tensor(np.concatenate([aa.detach().numpy() for aa in scores_sampmodel]))

    scores_sum = scores_sampmodel + scores
    # ======== FOR EACH TEST CHARACTER, GET ITS SCORE 
    if plotON is True:
        plt.figure()
        plt.plot(scores.detach().numpy(), '-ok')
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
Ntrials = 1000
import pickle
charall = np.random.randint(0, 10, Ntrials)

saveInterval = 20 # save mod this number trials

# for w, f, u in zip([5, 5, 10, 10, 20, 20],[5, 10, 5, 10, 5, 10],[20, 20, 20, 20, 20, 20]):
for w, f, u in zip([20, 20],[5, 10],[20, 20]):
    scores_all = []
    scores_sampmodel_all = []
    scores_sum_all = []
    idx_all = []
    for i, cc in enumerate(charall):
        print(i)
        cc = np.array(cc)
        scores, scores_sampmodel, idx, scores_sum = \
        getClassScore(charSamp=cc, Nway=w, frontierSize=f, nUpdates=u, plotON=False)

        scores_all.append(scores.detach())
        scores_sampmodel_all.append(scores_sampmodel.detach())
        scores_sum_all.append(scores_sum.detach())
        idx_all.append(idx)

        if i%saveInterval==0 and i>0:
            ## ========= SAVE OUTPUT
            datall = [scores_all, scores_sampmodel, scores_sampmodel_all, idx_all]

            # ======== SAVE
            with open('./datsave/datall_w%sf%su%s' %(w, f, u), 'wb') as fl:
                pickle.dump(datall, fl)

            print('==== SAVED!')
                
                
        ## ========= SAVE FINAL
        datall = [scores_all, scores_sampmodel, scores_sampmodel_all, idx_all]
        with open('./datsave/datall_w%sf%su%s' %(w, f, u), 'wb') as fl:
            pickle.dump(datall, fl)

