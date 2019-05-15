import os
import os.path
os.system("om-nodeinfo")

import argparse
import math
import pickle
import sys
import time
import shutil

import torch
from torch import optim
import numpy as np

from helmholtz import WSR, RWS, WS, IWAE, VAE
from aggregate import aggregate

pre_parser = argparse.ArgumentParser()
pre_parser.add_argument('--domain', choices=['cellular', 'regex', 'mnist'], default='mnist')
pre_args, other_args = pre_parser.parse_known_args()
domain = pre_args.domain

if domain=="cellular": import examples.cellular as M
elif domain=="regex": import examples.regex as M
elif domain=="mnist": import examples.mnist as M
else: raise NotImplementedError()

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
    if args.evaluate=="slurm" and args.evaluate_checkpoint is None:
        ckpt = "./checkpoint-%d.p" % model.iteration
        save(ckpt)
        slurmcmd="bash -c 'p=$(./slurmparams) sbatch $p --time=240 -J evaluate -o evaluate_" + str(model.iteration) + ".out om-run python " + " ".join(sys.argv) + " --evaluate-checkpoint=" + ckpt + "'"
        print(slurmcmd)
        os.system(slurmcmd)
    else:
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
        aggregate()
        #model.history.append(ev)
        return ev

lastsave = time.time()
def save(filename="./model.p"):
    global lastsave
    print("Saving")
    if os.path.isfile(filename): shutil.copyfile(filename, filename + ".bak")
    model.optstate = optimiser.state_dict()
    try: torch.save(model, filename + ".new")
    except: raise Exception("Couldn't save...")
    shutil.move(filename + ".new", filename)
    lastsave = time.time()








model = None
if args.evaluate_checkpoint is not None:
    model=torch.load(args.evaluate_checkpoint)
    print("Loaded %s" % args.evaluate_checkpoint)
else:
    try:
        model=torch.load("./model.p")
        print("Loaded model.p")
        if args.cuda: model = model.cuda()
        if not hasattr(model, 'pretraining'): model.pretraining=False
    except FileNotFoundError:
        pass 

    if model is None:
        print("Creating new model")
        px, pc, rc = M.Px(), M.Pc(), M.Rc()

        if args.load_pretrained is not None: raise NotImplementedError()

        pretrained = None
        if args.pretrain:
            try:
                pretrained=torch.load(pretrained_file)
                print("Loaded", pretrained_file)
                if hasattr(pretrained, "encoder"): 
                    rc=pretrained.encoder
                    print("-encoder")
                if hasattr(pretrained, "decoder"):
                    px=pretrained.decoder
                    print("-decoder")
                if hasattr(pretrained, "prior"):
                    pc=pretrained.prior
                    print("-prior")
            except FileNotFoundError:
                pass

        if args.mode == "WSR":
            model = WSR(rc, px, pc, nTasks=len(M.data), frontierSize=args.k, nUpdates=args.n_updates, rObjective=args.r_objective)
        elif args.mode == "prior":
            model = WSR(None, px, pc, nTasks=len(M.data), frontierSize=args.k, nUpdates=args.n_updates)
        #elif args.mode == "wsvae" or args.mode=="amwsvae": 
        #    model = WSVAE(rc, px, pc, qc, nTasks=len(M.data), ratio_threshold=args.ratio_threshold)
        elif args.mode == "VAE" or (args.mode=="IWAE" and args.k==1):
            model = VAE(rc, px, pc, nTasks=len(M.data))
        elif args.mode == "IWAE":
            model = IWAE(rc, px, pc, nTasks=len(M.data), nImportanceSamples=args.k)
        elif args.mode == "WS" or (args.mode=="RWS" and args.k==1):
            if args.mode=="RWS": assert args.r_objective=="sleep"
            model = WS(rc, px, pc, nTasks=len(M.data))
        elif args.mode == "RWS":
            model = RWS(rc, px, pc, nTasks=len(M.data), nImportanceSamples=args.k, rObjective=args.r_objective)

        if args.cuda:
            model = model.cuda()

        if pretrained is None:
            model.iteration = 0
            model.wallclock = 0
        else:
            model.iteration = pretrained.iteration
            model.wallclock = pretrained.wallclock

        model.pretraining=args.pretrain>0
        model.last_eval_time=None
        model.optstate=None

        model.batch_size=args.batch_size
        model.pretrain_batch_size=args.pretrain_batch_size
        model.consecutive_errors=0


if args.command is not None:
    if args.command=="100way":
        classification_100_way, _ = getClassification(100, args.classification_samples)
        print("New 100-way Accuracy:", "%5.2f" % (classification_100_way*100) + "%", flush=True)
    elif args.command=="20way":
        classification_20_way, _ = getClassification(20, args.classification_samples)
        print("New 20-way Accuracy:", "%5.2f" % (classification_20_way*100) + "%", flush=True)
    elif args.command=="derivations":
        M.showDerivations(model, 100, save=True)
    else:
        getattr(M, args.command)(model)
elif args.evaluate_checkpoint is not None:
    evaluate()
else:
    # Training
    if args.optimiser=="sgd": opt=optim.SGD
    if args.optimiser=="adam": opt=optim.Adam

    def make_optimiser():
        lr = args.pretrain_lr if model.pretraining else args.lr
        if hasattr(model, "baseline_bias"):
            normal_params = [x for x in model.parameters() if id(x) not in [id(model.baseline_bias), id(model.baseline_task)]]
            optimiser = opt([
                {'params': normal_params, 'lr':lr},
                {'params': [model.baseline_bias, model.baseline_task], 'lr':1}
            ])
        else:
            optimiser = opt(model.parameters(), lr=lr)
        return optimiser
    optimiser = make_optimiser()
    if model.optstate is not None: optimiser.load_state_dict(model.optstate)
    #optimiser = optim.SGD(model.parameters(), lr=1e-2)


    if args.hyperprior_weight != 0 and not hasattr(model.prior, "hyperprior"):
        print("Warning: model does not have a hyperprior!")

    pretrain_avg = None
    def pretrainStep():
        global pretrain_avg
        optimiser.zero_grad()
        i, c, x = M.getPretrainBatch(model)
        _, priorscore = model.prior(i, c)
        #_, likelihood = model.decoder(i, c, x)
        _, encscore = model.encoder(i, x, c)
        objective = priorscore.mean() + encscore.mean() #+ likelihood.mean()
        (-objective).backward()
        optimiser.step()
        pretrain_avg = objective.item() if pretrain_avg is None else objective.item()*0.03 + pretrain_avg*0.97
        if model.iteration%10==0:
            print("Pretraining Iteration", model.iteration, "Score %3.3f" % pretrain_avg, flush=True)
            c, _ = model.prior(i)
            print("Prior:", c[0])
            c, _ = model.encoder(i, x)
            print("Encoder:", x[0], "--->", c[0])
            print()

    def trainStep():
        optimiser.zero_grad()
        if hasattr(model.decoder, 'ready'): model.decoder.ready()
        starttime = time.time()
        idxs, xs = getBatch(model.batch_size)
        annealing=getAnnealing(model.iteration)
        objective, kl = model(idxs, xs, return_kl=True, annealing=annealing)
        objective = objective.mean()
        kl = kl.mean()
        
        if args.hyperprior_weight != 0:
            if hasattr(model.prior, 'hyperprior'):
                hyperprior = model.prior.hyperprior()
                if hyperprior is not None:
                    hyperprior_score = hyperprior*args.hyperprior_weight * args.batch_size / len(M.data)
                    objective = objective + hyperprior_score - hyperprior_score.data
            else:
                hyperprior = None
        else:
            hyperprior=None

        (-objective).backward()
        optimiser.step()
        model.wallclock += time.time() - starttime
        if model.iteration<20 or model.iteration%10==0:
            print("Iteration", model.iteration,
                 "Score %3.3f" % objective.item(),
                 "" if hyperprior is None else "Hyperprior %3.3f" % hyperprior.item(),
                 "KL %3.3f" % kl.item(),
                 "(anneal %3.3f)" % annealing,
                 flush=True)

    while (model.pretraining or model.iteration<=args.iterations) or model.wallclock<args.minutes*60:
        try:
            if args.evaluate != "none" and not model.pretraining and (
                    model.last_eval_time is None or 
                   (model.last_eval_time<args.evaluate_minutes*60 and model.wallclock>model.last_eval_time*2+120) or 
                   model.wallclock > model.last_eval_time+60*args.evaluate_minutes):
                torch.cuda.empty_cache()
                evaluate()
                model.last_eval_time=model.wallclock

            #if model.iteration%100==0 and args.auto_batchsize:
            #    model.batch_size = 1+int(model.batch_size*1.05)
            #    print("Increasing batch size to:", model.batch_size)

            if model.pretraining:
                pretrainStep()
                model.iteration += 1
                if model.iteration%1000==0 and not args.no_save:
                    print("saving...")
                    torch.save(model, pretrained_file)
                if model.iteration >0 and model.pretraining: 
                    model.iteration = 0
                    model.pretraining = False
                    optimiser = make_optimiser()
            else:
                trainStep()
                model.iteration += 1
                M.iteration(model)

            model.consecutive_errors = 0


            if time.time() - lastsave > 1200 and not args.no_save: #save every 20 minutes
                save()
        except RuntimeError as err:
            if "cuda" in err.args[0].lower() and "out of memory" in err.args[0].lower():
                model.consecutive_errors += 1
                if model.consecutive_errors == 5: #20:
                    print(model.consecutive_errors, "consecutive errors. Quitting.")
                    raise err
                else:
                    torch.cuda.empty_cache()
                    print(err)
                    print("\nResuming...\n")
                    if model.pretraining and args.auto_batchsize:
                        model.batch_size = int(model.batch_size*0.9)
                        print("Decreasing batch size to:", model.batch_size)
            else:
                raise err


    print("Complete")
    if hasattr(M, 'complete'):
        M.complete(model)
    if not args.no_save: save()
    evaluate()
