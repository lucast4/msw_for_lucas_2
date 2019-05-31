#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:22:05 2019

Visualization of motor programs
(TO ADD) Visualization of latent variables

@author: lucastian
"""

subtractMean = False # subtract mean motor program (across samples)
testOrTrain='Train';  # NOT DONE YET !!! TODO

## ********************

import torch
import matplotlib.pyplot as plt
import numpy as np

nContentLatent = 90 # verified in mnist.py code


## *************** LOAD MODEL
model=torch.load("./model.p", map_location='cpu') #LT
print("Loaded model.p")



## *************** LOAD TRAIN AND TEST DATASETS
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





# =============================================================================
# RUN            
# =============================================================================
            
## ******** get overall mean motor program - to subtract if desires
cs, _ = model.sample(range(len(M.data)), x=M.data, noise=0, sample_probs=True)
cs = np.array(np.array(cs))
csmean = cs.mean(axis=0)
            
    
## ********* EXTRACT STROKE IDs
nStrokes = model.decoder.px.nStrokes
nSpatialLatent = 0
binary = model.decoder.px.binary
noise = 0

c, x_all = model.sample(range(len(M.data)), x=M.data, noise=0, sample_probs=True)
c_tens = model.t.new_tensor(c)#.reshape(len(c), args.nLatent)
c_content = c_tens[:, :nContentLatent]
c_on = c_content[:, :nStrokes]
c_on[:,0] = 0 # first stroke is always off.
c_lines = c_content[:, nStrokes:]
if nSpatialLatent>0:
    c_spatial = c_tens[:, nContentLatent:]
c_idx = (c_lines.view(len(c_tens), nStrokes, len(binary)) * binary.data[None, None, :]).sum(dim=2).long()
z = model.t.new_zeros(len(c_idx), 1, 1, 1)
dx = model.decoder.px.dx[c_idx]
dy = model.decoder.px.dy[c_idx]
x1 = torch.cat([z, dx[:, :-1].cumsum(dim=1)], dim=1) + 0.5
y1 = torch.cat([z, dy[:, :-1].cumsum(dim=1)], dim=1) + 0.5
lines = model.decoder.px.get_lines(c_idx.view(-1), x1.view(-1), y1.view(-1), noise=noise)
lines = lines.reshape(-1, nStrokes, lines.shape[1], lines.shape[2])




## *************** FIND ALL CASES OF A GIVEN CHARACTER AND PLOT
def plotRaw(charthis):
    idxs = [i for i, ll in enumerate(M.datalabels) if ll==charthis]
    if len(idxs)>10:
        idxs = np.random.choice(idxs, size=10)
    
    # --- SAMPLE
    cs, xs = model.sample(idxs, x=M.data[idxs], noise=0, sample_probs=True)
    #xs = [xxss.detach().numpy() for xxss in xs]
    
    # ---- plot all
    plt.figure(figsize=(10, 20))
    for i, x in enumerate(xs):
        plt.subplot(1, 20, i+1)
        plt.imshow(x[0].detach().numpy())
    
    # ======== PLOT HEAT MAP OF ALL MOTOR PROGRAM
    cs = np.array(np.array(cs))
    if subtractMean is True:
        cs = cs-csmean
    plt.figure(figsize=(10, 20))
    plt.imshow(cs, cmap='PuOr', vmin=-1, vmax=1)
    #plt.colorbar()
    plt.plot([9.5, 9.5], [cs.shape[0]-1, 0], Color='r')
    plt.title('latent motor program, char %s' %charthis)
    plt.xlabel('sequence position')
    plt.ylabel('sample #')
    for xx in np.linspace(17.5, 89.5, 10):
        plt.plot([xx, xx], [cs.shape[0]-1, 0], Color='r')
        
    # ========== PLOT lines for index for each stroke
    plt.figure()
    for cc in c_idx[idxs]:
        plt.plot(range(len(cc)), cc.numpy(), '-k')
        
    for cc_on, cc in zip(c_on[idxs], c_idx[idxs]):
        xthis = np.argwhere(cc_on.numpy()).squeeze()
        xthis = xthis[np.argwhere(xthis>0)]
        cthis = cc[xthis.reshape(-1,)].numpy()
        plt.plot(xthis, cthis, 'or')
    plt.title('base10 index for each stroke')
    plt.ylabel('stroke index')
    plt.xlabel('ordinal position')
            
    # ======== PLOT the distribution of stroke numbers used
    plt.figure()   
    tmp = []
    for cc_on, cc in zip(c_on[idxs], c_idx[idxs]):
        xthis = np.argwhere(cc_on.numpy()).squeeze()
        xthis = xthis[np.argwhere(xthis>0)]
        cthis = cc[xthis.reshape(-1,)].numpy()        
        tmp.append(cthis)
    plt.hist(tmp)
    plt.title('histogram of visible strokes (color=sample)')
    plt.xlabel('stroke id')
    plt.ylabel('count')    
    
    # ======= PLOT STROKE SEQUENCES
    for i in idxs:
        ll = lines[i]
        m_orig = M.data[i][0].detach().numpy()
        m_gen = x_all[i][0].detach().numpy()
        c_on_this = c_on[i]
        plt.figure(figsize=(10, 20))
        # -- plot the final
        plt.subplot(1, nStrokes+2, 1)
        plt.imshow(m_orig, vmin=0, vmax=1)
        plt.subplot(1, nStrokes+2, 2)
        plt.imshow(m_gen, vmin=0, vmax=1)
        for ii, lll in enumerate(ll):
            plt.subplot(1, nStrokes+2, ii+3)
            plt.imshow(lll.detach().numpy(), vmin=0, vmax=1)
            if c_on_this[ii]==1 and ii>0:
                plt.title('ON',color='g')
            else:
                plt.title('OFF',color='r')
            
            
            

def plotEmbeddings():
    ## ******** tsne embedding of programs (l80, ignore whether on or off)
    from sklearn.manifold import TSNE as tsne
    
    for i, p in enumerate((2,5,10,15,20,30,40,50)):
        X = tsne(perplexity=p).fit_transform(c_lines)
        
        plt.figure(figsize=(6, 8))
        plt.title('(tsne on binary encoding (l-80)) perpl=%s' %p)
        for j in range(10):
            idxs = np.argwhere(np.array(M.datalabels)==j)
            plt.plot(X[idxs,0], X[idxs, 1], 'o', label=j)
            plt.legend(loc=1)
            
    
    ## **** tsne on stroke indices 
    if False:
        # skip this since it also includes OFF strokes
        # (i.e. each trial is a l-256 binary string)
        
        #enc = OneHotEncoder(sparse=False, categorical_features=range(256))
        #X_enc = enc.fit_transform(X = c_idx[:,0].reshape(-1,1))
            
        X_enc, _ = zip(*[np.histogram(bins=np.linspace(-0.5, 255.5, 257), a=c_idx[jj]) for jj in range(len(c_idx))])
        X_enc = np.asarray(X_enc)
        
        for i, p in enumerate((2,5,10,15,20,30,40,50)):
            X = tsne(perplexity=p).fit_transform(X_enc)
            
            plt.figure(figsize=(6, 8))
            plt.title('(tsne on strokes (l-10, including off) perpl=%s' %p)
            for j in range(10):
                idxs = np.argwhere(np.array(M.datalabels)==j)
                plt.plot(X[idxs,0], X[idxs, 1], 'o', label=j)
                plt.legend(loc=1)
            
    ## **** tsne on stroke indices [only ON strokes]
    X_enc, _ = zip(*[np.histogram(bins=np.linspace(-0.5, 255.5, 257), a=c_idx[jj, np.argwhere(np.logical_and(c_on[jj]==1, jj>0))]) for jj in range(len(c_idx))])
    X_enc = np.asarray(X_enc)
    
    
    for i, p in enumerate((2,5,10,15,20,30,40,50)):
        X = tsne(perplexity=p).fit_transform(X_enc)
        
        plt.figure(figsize=(6, 8))
        plt.title('(tsne on strokes (l-10))[ONLY ON] perpl=%s' %p)
        for j in range(10):
            idxs = np.argwhere(np.array(M.datalabels)==j)
            plt.plot(X[idxs,0], X[idxs, 1], 'o', label=j)
            plt.legend(loc=1)
            
         
# =============================================================================
# LIST ALL STROKES
# =============================================================================
# --- 1) get all unique stroke IDs
#tmp = [c_idx[:,1:], lines[:,1:], c_on[:, 1:]]

#c_idx[inds] for np.argwhere(c_on[j]==1) for j in len(c_idx)]

if __name__ == "__main__":
    c_on_flat = c_on.reshape(-1).detach().numpy()         
    c_idx_flat = c_idx.reshape(-1).detach().numpy()
    lines_flat = lines.reshape(-1, 28, 28).detach().numpy()
    
    _, tmp = np.unique(c_idx_flat, return_index=True)
    c_idx_U = c_idx_flat[tmp]
    lines_U = lines_flat[tmp]
    
    # ==== plot all strokes
    nrow = np.ceil(len(c_idx_U)/20)
    ax = plt.figure(figsize=(20, 30))
    for i, (ind, stroke) in enumerate(zip(c_idx_U, lines_U)):
        plt.subplot(nrow, 20, i+1)
        plt.imshow(stroke, vmin=0, vmax=1)
        plt.title('#%s' % ind)
                  
    
    # ******* PLOT RAW
    plotRaw(1)
    
    # ******* PLOT EMBEDDINGS
    plotEmbeddings()
