import os
import os.path
os.system("om-nodeinfo")

import argparse
import math
import pickle
import time

import torch
import numpy as np



import examples.mnist as M

parser = argparse.ArgumentParser()
parser.add_argument('--domain')
parser.add_argument('--command', type=str, default=None)
# Which learning algorithm
parser.add_argument('--mode', type=str, choices=['VAE', 'IWAE', 'WS', 'RWS', 'WSR', 'PRIOR'], default="WSR")
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--r_objective', choices=['wake', 'sleep', 'mix'], default='wake')
# Hyperparams
parser.add_argument('--anneal', type=int, default=0) 
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--pretrain_batch_size', type=int, default=250)
parser.add_argument('--optimiser', choices=['adam', 'sgd'], default='sgd') 
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--pretrain_lr', type=float, default=0.02)
parser.add_argument('--hyperprior_weight', type=float, default=1)
parser.add_argument('--ratio_threshold', type=float, default=1)
parser.add_argument('--n_updates', type=int, default=1)
# Evaluation
parser.add_argument('--generate_every', type=int, default=50) 
parser.add_argument('--evaluate-checkpoint', type=str, default=None) 
parser.add_argument('--evaluate', choices=['slurm', 'serial', 'none'], default='serial')
parser.add_argument('--evaluate_minutes', type=int, default=30)
parser.add_argument('--classification_samples', type=int, default=100)
parser.add_argument('--elbo_samples', type=int, default=100)
# General
parser.add_argument('--no-save', dest='no_save', action='store_const', const=True, default=False)
parser.add_argument('--no-cuda', dest='cuda', action='store_const', const=False, default=True)
parser.add_argument('--iterations', type=int, default=10000)
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--auto-batchsize', dest='auto_batchsize', action='store_const', const=True, default=False)
parser.add_argument('--load_pretrained', type=str, default=None)
parser.add_argument('--evaluate_checkpoint', type=str, default=None) 
parser.add_argument('--minutes', type=int, default=0)
args = parser.parse_args(M.unknown_args)


if args.pretrain != 0:
    os.makedirs("./pretrained", exist_ok=True)
    pretrained_file = "./pretrained/" + M.argstr() + "_" + str(args.pretrain_lr) + "_" + str(args.pretrain_batch_size) + ".p"

def getBatch(batch_size):
    idxs = np.random.choice(len(M.data), size=batch_size)
    return idxs, [M.data[i] for i in idxs]

def getAnnealing(iteration):
    if args.anneal ==0 or iteration>args.anneal: return 0
    else:
        x = (iteration-args.anneal/2)/(args.anneal/20)
        if x<-100: return 1
        elif x<100: return 1 / (1 + math.exp(x))
        else: return 0

def getClassification(n_way=100, n_samples=1000):
    n_samples = min(n_samples, len(M.data))
    n_way = min(n_way, len(M.data))
    print("Evaluating %d-way classification accuracy"%n_way, flush=True)
    starttime=time.time()
    #i = list(range(len(testM.data)))
    hits = 0
    total = 0
    predictive = 0
    for trueclass in range(n_samples): #100 classes np.random.choice(range(len(testM.data)), size=500, replace=False):
        x = M.testdata[trueclass]
        others = list(range(trueclass)) + list(range(trueclass+1, len(M.testdata)))
        i_rearranged = [trueclass] + list(np.random.choice(others, size=n_way-1, replace=False))
        x_inp = [M.data[ii] for ii in i_rearranged]
        scores = model.conditional(i_rearranged, x_inp, [x]*n_way)
        predictive += scores[0].mean().item() #.item() so we don't hold onto tensor for gradient information
        #print("scores:", scores)
        if not (scores>scores[0]).any():
            num_tied = (scores==scores[0]).sum().item()
            hits += 1/num_tied 
        total += 1
    print("Took", int(time.time()-starttime), "seconds")
    return hits / total, predictive/total


def getELBo():
    print("Evaluating ELBo", flush=True)
    elbo = None
    kl = None
    n_runs=args.elbo_samples
    for i in range(n_runs):
        idxs, xs = getBatch(model.batch_size)
        _elbo, _kl = model.elbo(idxs, xs, return_kl=True)
        if elbo is None:
            elbo=_elbo.mean().item() #.item() so we don't hold onto tensor for gradient info
            kl=_kl.mean().item()
        else:
            elbo += _elbo.mean().item()
            kl += _kl.mean().item()
    return elbo/n_runs, kl/n_runs

def evaluate():
    M.evaluate(model)
    evaluate_start=time.time()
    elbo, kl = getELBo()
    print("elbo:", elbo)
    print("KL:", kl)
    n_classification_samples = args.classification_samples
    classification_20_way, predictive = getClassification(20, n_classification_samples)
    print("20-way Accuracy:", "%5.2f" % (classification_20_way*100) + "%", flush=True)
    precision_20_way = math.sqrt(classification_20_way * (1-classification_20_way) / n_classification_samples)
    #classification_100_way, _ = getClassification(100, n_classification_samples)
    #print("100-way Accuracy:", "%5.2f" % (classification_100_way*100) + "%", flush=True)
    #precision_100_way = math.sqrt(classification_100_way * (1-classification_100_way) / n_classification_samples)
    print("Evaluate took a total of:", int((time.time()-evaluate_start)/60), "minutes")
    ev = {'20-way':classification_20_way,
            #'100-way':classification_100_way,
            'precision-20-way':precision_20_way,
            #'precision-100-way':precision_100_way,
            #'predictive':predictive,
            'ELBo':elbo,
            'kl':kl,
            'time':model.wallclock,
            'iteration':model.iteration}
    with open("evaluate_%d.p" % model.iteration, "wb") as f:
        pickle.dump(ev, f)
    #aggregate()
    #model.history.append(ev)
    return ev



model=torch.load("./model.p")
print("Loaded model.p")
if args.cuda: model = model.cuda()



############## To make noisy samples #########################3
print("Making originals")
for i, x_data in enumerate(M.data[:100]):
    M.imFromTensor(x_data[0]).save("results/%d-original.png" % i)

print("Making noisy")
for j in range(5):
    print(j)
    cs, xs = model.sample(i=np.arange(100), x=M.data[:100], noise=0.25, sample_probs=True)
    for i, (x_data, x_sample) in enumerate(zip(M.data, xs)):
        M.imFromTensor(x_sample[0]).save("results/" + str(i) + "-sample" + str(j) + ".png")


#### To get conditional log probability ####
for j in range(10):
    print("Conditional log_probs for test example", j)
    M.imFromTensor(M.testdata[j, 0]).save("results/test-%d.png" % j)
    log_probs = model(i=np.arange(100), x=M.testdata[j:j+1].repeat(100, 1, 1, 1))
    print(log_probs)
