#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:10:22 2019

@author: lucastian
"""
import numpy as np
import copy


def flipOnOffNoScore(model, c, indchange):
    # ============ Flip whether a stroke is on or off
    nStrokes = model.decoder.px.nStrokes
    cc = model.t.new_tensor(c)#.reshape(len(c), args.nLatent)
    
    #indchange = 0
    assert(indchange<=nStrokes)
    cc[:, indchange] = 1-cc[:, indchange]
    
    return cc

    
def flipOnOff(model, i, c, x, indchange):
    # indchange, which stroke to flip on/off.
    
    # ============ Flip whether a stroke is on or off
    nStrokes = model.decoder.px.nStrokes
 #   cc = model.t.new_tensor(c)#.reshape(len(c), args.nLatent)
    
    #indchange = 0
    assert(indchange<=nStrokes)
#    cc[:, indchange] = 1-cc[:, indchange]
    
    corig = copy.deepcopy(c)
    #print(c[0][0])
    for j in range(len(c)):
        for jj in range(len(c[j])):
            #print(c[j][jj])
            corig[j][jj][indchange] = 1 - c[j][jj][indchange]
            #print(c[j][jj])
    #print(corig[0][0])
    #print(c[0][0])  
                  
    # ========= calculate score
    #c = c.tolist()
    score_all = []
    #cout = []
    for j in range(len(corig)):
        _, priorscore = model.prior(i[j], corig[j])
        _, likelihood = model.decoder([i[j] for _ in range(len(corig[j]))], corig[j], [x[j] for _ in range(len(corig[j]))])
#_, priorscore = model.prior(i, c)
#    _, likelihood = model.decoder(i, c, )
        
        score_all.append(priorscore + likelihood)
        #cout.append(c[j])
    #score = (priorscore + likelihood).tolist()

    return corig, score_all


def makeUpdateMCMC(model, i, x, strokeNum, verbose=False):
    # Fills any unfilled mixture compoentns and generate proposals by flipping 
    # on(if off) or off(if on) the stroke at strokeNum slot. Each mixture component
    # contributes one proposal. They are then all thrown into a pool with the 
    # existing components, and the top N (frontier size) are kept.
    batch_size = len(x)
    task_update_data = {}

    unfilled_idxs = {idx:model.frontierSize - len(model.mixtureScores[i[idx]])
                        for idx in range(batch_size) if len(model.mixtureScores[i[idx]]) < model.frontierSize}
    if len(unfilled_idxs)>0:
        unfilled_i = [i[idx] for idx,num_repeats in unfilled_idxs.items() for _ in range(num_repeats)]
        unfilled_x = [x[idx] for idx,num_repeats in unfilled_idxs.items() for _ in range(num_repeats)]
        unfilled_c, _ = model.encoder(i, x) if model.encoder is not None else model.prior(i)
        for ii,cc,xx in zip(unfilled_i, unfilled_c, unfilled_x):
            if cc not in model.mixtureComponents[ii]:
                model.mixtureComponents[ii].append(cc)
                task_update_data[ii]=xx
    
        
    # ******************* make new proposals
    # indchange = 0 # which stroke number to flip?
    c = [model.mixtureComponents[ii] for ii in i]
    cc, score = flipOnOff(model, i, c, x, strokeNum)
    #assert c[0][0] not in cc[0]
    
    # ===== update if score is better than current components
    #c, _ = model.encoder(i, x) if model.encoder is not None else model.prior(i)        
    #_, priorscore = model.prior(i, c)
    #_, likelihood = model.decoder(i, c, x)
    #score = (priorscore + likelihood).tolist()
    for idx in range(batch_size):
        if verbose:
            print('proposal:')
            print(score[idx])
            print('previous:')
            print(model.mixtureScores[i[idx]])
        for jj in range(len(cc[idx])):
            scorethis = score[idx][jj]
            ccthis = cc[idx][jj]
            if scorethis > min(model.mixtureScores[i[idx]]) and ccthis not in model.mixtureComponents[i[idx]]:
                print('MCMC proposal added! char%s, mcomponent%s' %(i[idx], jj))
                model.mixtureComponents[i[idx]].append(ccthis)
                task_update_data[i[idx]] = x[idx]

    # ===== add any new components and then remove lowest components
    for ii,xx in task_update_data.items():
        _, priorscores = model.prior([ii for _ in model.mixtureComponents[ii]],
                                    model.mixtureComponents[ii])
        _, likelihoods = model.decoder([ii for _ in model.mixtureComponents[ii]],
                                      model.mixtureComponents[ii],
                                      [xx for _ in range(len(model.mixtureComponents[ii]))])
        model.mixtureScores[ii] = (priorscores+likelihoods).tolist()
        while len(model.mixtureComponents[ii]) > model.frontierSize:
            min_idx = np.argmin(model.mixtureScores[ii])
            model.mixtureComponents[ii] = model.mixtureComponents[ii][:min_idx] + model.mixtureComponents[ii][min_idx+1:]
            model.mixtureScores[ii] = model.mixtureScores[ii][:min_idx] + model.mixtureScores[ii][min_idx+1:]
        model.mixtureWeights.data[ii][:len(model.mixtureScores[ii])] = model.t.new(model.mixtureScores[ii])
        model.mixtureWeights.data[ii][len(model.mixtureScores[ii]):] = float("-inf")
        model.nMixtureComponents[ii] = len(model.mixtureComponents[ii])    

    return model