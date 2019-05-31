from builtins import super

import numpy as np
import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from MCMC import flipOnOff

def nograd(x):
    return x.data if torch.is_tensor(x) else x

class EncoderDecoder(nn.Module):
    @property
    def amortised_network(self):
        if hasattr(self, 'encoder'):
             return self.encoder
        else:
             return None

    def conditional(self, i, D, x, n_samples=5, f=None):
        if f is None: f = lambda c: self.decoder(i, c, x)[1]
        scores = self.t.new_zeros([n_samples, len(x)])
        for sample_idx in range(n_samples):
            if self.amortised_network: c, encscore = self.amortised_network(i, D, subsample=False)
            else: c, encscore = self.prior(i) 
            likelihood = f(c)
            scores[sample_idx] = likelihood
        return scores.logsumexp(dim=0) - math.log(n_samples)

    def sample(self, i, x):
        if self.amortised_network: c, _ = self.amortised_network(i, x)
        else: c, _ = self.prior(i)
        x, _ = self.decoder(i, c)
        return c, x

    def sample_prior(self, i):
        c, _ = self.prior(i)
        x, _ = self.decoder(i, c)
        return c, x

    def elbo(self, i, x, return_kl=False):
        if self.amortised_network: c, encscore = self.amortised_network(i, x)
        else: c, encscore = self.prior(i)
        _, priorscore = self.prior(i, c)
        _, likelihood = self.decoder(i, c, x)
        kl = encscore - priorscore
        elbo = likelihood - kl
        if return_kl:
            return elbo, kl
        else:
            return elbo

    def marginal(self, i, x, return_kl=False, k=100, print_every=20):
        assert self.amortised_network
        log_ws = []
        def make():
            logws = torch.cat(log_ws, dim=1)
            return logws.logsumexp(dim=1) - math.log(len(log_ws))

        c0 = []
        for samp in range(k):
            c, encscore = self.amortised_network(i,x)
            #print(encscore[0])
            c0.append(tuple(c[0]))
            _, priorscore = self.prior(i, c)
            _, likelihood = self.decoder(i, c, x)
            log_ws.append((priorscore + likelihood - encscore).unsqueeze(1).detach().cpu())
            if samp==0 or samp%(print_every)==print_every-1:
                print("Sample", (samp+1), "/", k, " estimate", make().mean().item(), "(%d unique cs)" % len(set(c0))) 
        return make()
        


class WSR(EncoderDecoder):
    def __init__(self, encoder, decoder, prior, nTasks, frontierSize=10, nUpdates=1, rObjective="sleep"):
        """
        If encoder is None, sample from prior
        rObjective: "wake" or "sleep" or "mix"
        """
        super().__init__()
        self.nTasks = nTasks
        self.frontierSize = frontierSize
        self.nUpdates = nUpdates
        self.rObjective = rObjective

        self.mixtureComponents = [[] for _ in range(nTasks)]
        self.mixtureWeights = Parameter(torch.zeros(nTasks, frontierSize)) #Unnormalised log-q
        self.mixtureScores = [[] for _ in range(nTasks)] #most recent log joint
        self.nMixtureComponents = Parameter(torch.zeros(nTasks))
        self.t = Parameter(torch.zeros(1))

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def getDist(self, i):
        weights = self.mixtureWeights[i]
        return Categorical(F.softmax(weights, dim=1))

    def sample(self, i, x, **decoder_args):
        if i is None: return super().sample(i, x)

        ix_update = [(ii,xx) for ii,xx in zip(i,x) if len(self.mixtureComponents[ii]) == 0]
        if len(ix_update)>0: 
            i_update = [ii for ii,xx in ix_update]
            x_update = [xx for ii,xx in ix_update]
            self.makeUpdates(i_update, x_update, 1)

        idxs = self.getDist(i).sample()
        idxs = idxs.tolist()
        c = [self.mixtureComponents[ii][idx] for ii,idx in zip(i, idxs)]
        x, _ = self.decoder(i, c, **decoder_args)
        return c, x

    def conditional(self, i, D, x, n_samples=None, f=None, noise=0):
        if i is None : return super().conditional(i, D, x, f=f)
        if n_samples is not None: return super().conditional(i, D, x, n_samples, f)
        if f is None: f = lambda c: self.decoder(i, c, x, noise=noise)[1]
        #print("Running the real conditional!")
        self.ensure_nonempty(i, D)
        logprobs = F.log_softmax(self.mixtureWeights[i], dim=1) # makes the max 0, rest are neg from that.
        scores = self.t.new_full([self.frontierSize, len(i)], float("-inf"))
        for component_idx in range(self.frontierSize):
            i_has_component = self.nMixtureComponents[i]>component_idx 
            lp = logprobs[:, component_idx].masked_select(i_has_component)
            c = [self.mixtureComponents[ii][component_idx] for ii,has_component in zip(i, i_has_component) if has_component]
            # if len(c)>0:
            if len(c)==len(D): # LT MODIFIED - so that only continues if all cases in D have and inferred c for this componet index (i.e mixture compnent)
                likelihood = f(c)
                scores[component_idx].masked_scatter_(i_has_component, lp + likelihood)
        return scores.logsumexp(dim=0) # takes sum across all components (i.e. one sum for each datapoint). TODO: some case has -Inf score for some component...

    def getNovelPriorSamples(self, n=500):
        c, _ = self.prior([None]*n)
        used = [v for component in self.mixtureComponents for v in component]
        novel = list(set(c) - set(used))
        proportionNovel = len(novel)/n
        return novel, proportionNovel

    def makeUpdates(self, i, x, nUpdates):
        batch_size = len(x)
        task_update_data = {}

        unfilled_idxs = {idx:self.frontierSize - len(self.mixtureScores[i[idx]])
                            for idx in range(batch_size) if len(self.mixtureScores[i[idx]]) < self.frontierSize}
        if len(unfilled_idxs)>0:
            unfilled_i = [i[idx] for idx,num_repeats in unfilled_idxs.items() for _ in range(num_repeats)]
            unfilled_x = [x[idx] for idx,num_repeats in unfilled_idxs.items() for _ in range(num_repeats)]
            unfilled_c, _ = self.encoder(i, x) if self.encoder is not None else self.prior(i)
            for ii,cc,xx in zip(unfilled_i, unfilled_c, unfilled_x):
                if cc not in self.mixtureComponents[ii]:
                    self.mixtureComponents[ii].append(cc)
                    task_update_data[ii]=xx
            
        for iUpdate in range(nUpdates):
            c, _ = self.encoder(i, x) if self.encoder is not None else self.prior(i)        
            _, priorscore = self.prior(i, c)
            _, likelihood = self.decoder(i, c, x)
            score = (priorscore + likelihood).tolist()
            for idx in range(batch_size):
                if (len(self.mixtureScores[i[idx]]) < self.frontierSize or score[idx] > min(self.mixtureScores[i[idx]])) \
                        and c[idx] not in self.mixtureComponents[i[idx]]:
                    self.mixtureComponents[i[idx]].append(c[idx])
                    task_update_data[i[idx]] = x[idx]

        for ii,xx in task_update_data.items():
            _, priorscores = self.prior([ii for _ in self.mixtureComponents[ii]],
                                        self.mixtureComponents[ii])
            _, likelihoods = self.decoder([ii for _ in self.mixtureComponents[ii]],
                                          self.mixtureComponents[ii],
                                          [xx for _ in range(len(self.mixtureComponents[ii]))])
            self.mixtureScores[ii] = (priorscores+likelihoods).tolist()
            while len(self.mixtureComponents[ii]) > self.frontierSize:
                min_idx = np.argmin(self.mixtureScores[ii])
                self.mixtureComponents[ii] = self.mixtureComponents[ii][:min_idx] + self.mixtureComponents[ii][min_idx+1:]
                self.mixtureScores[ii] = self.mixtureScores[ii][:min_idx] + self.mixtureScores[ii][min_idx+1:]
            self.mixtureWeights.data[ii][:len(self.mixtureScores[ii])] = self.t.new(self.mixtureScores[ii])
            self.mixtureWeights.data[ii][len(self.mixtureScores[ii]):] = float("-inf")
            self.nMixtureComponents[ii] = len(self.mixtureComponents[ii])

    def elbo(self, i, x, return_kl):
        if i is None: return super().elbo(i, x, return_kl)
        return self.forward(i, x, return_kl, fixed=True)

    def ensure_nonempty(self, i, x):
        ix_update = [(ii,xx) for ii,xx in zip(i,x) if len(self.mixtureComponents[ii]) == 0]
        if len(ix_update)>0: 
            i_update = [ii for ii,xx in ix_update]
            x_update = [xx for ii,xx in ix_update]
            self.makeUpdates(i_update, x_update, 1)

    def forward(self, i, x, return_kl=False, annealing=0, fixed=False, **decoder_args):
        # Maybe update mixture
        if not fixed:
            self.makeUpdates(i, x, self.nUpdates)
            if self.encoder is not None and (self.rObjective == "sleep" or self.rObjective == "mix"):
                # Sleep-R objective
                _c, _ = self.prior(i)
                _x, _ = self.decoder(i, _c)
                _, r_score = self.encoder(i, nograd(_x), _c)
            else:
                r_score = 0
        else:
            self.ensure_nonempty(i,x)
            r_score = 0

        # Wake
        dist = self.getDist(i) 
        try:
            j = dist.sample()
        except Exception:
            import pdb
            pdb.set_trace()
        c = [self.mixtureComponents[ii][jj] for ii,jj in zip(i,j.tolist())]

        # (Wake-phase R update)
        if self.rObjective == "wake" or self.rObjective=="mix":
            _, r_score_wake = self.encoder(i, x, c) 
            if self.rObjective == "wake": r_score = r_score_wake
            else: r_score = r_score*0.5 + r_score_wake*0.5

        # (Wake-phase G update)
        _, priorscore = self.prior(i, c)
        _, likelihood = self.decoder(i, c, x, **decoder_args)
        mixtureScores = priorscore + likelihood
        for idx,(ii,jj) in enumerate(zip(i,j)):
            self.mixtureScores[ii][jj] = mixtureScores[idx].item()

        lp = dist.log_prob(j)
        kl = lp - priorscore
        score = likelihood - (1-annealing)*kl
        p_score = score + score.data*lp
        
        # Total
        objective = p_score + r_score
        
        if annealing==0:
            objective = objective - objective.data + score.data
        else:
            objective = objective - objective.data + (likelihood - kl).data

        if return_kl:
            return objective, kl
        else:
            return objective

class IWAE(EncoderDecoder):
    def __init__(self, encoder, decoder, prior, nTasks, nImportanceSamples=1):
        super().__init__()
        self.nTasks = nTasks
        self.nImportanceSamples = nImportanceSamples

        self.t = Parameter(torch.zeros(1))

        self.baseline_bias = Parameter(torch.zeros(1))
        self.baseline_task = Parameter(torch.zeros(nTasks))

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, 'baseline_exp_pos'): #legacy
            self.baseline_task = self.baseline_exp_pos 
            del self.baseline_exp_pos
            del self.baseline_exp_neg

    def baseline_shared(self): return self.baseline_bias
    def baseline_residual(self): return self.baseline_task

    def forward(self, i, x, return_kl=False, annealing=0, fixed=False):
        batch_size = len(i)

        i = [ii for ii in i for _ in range(self.nImportanceSamples)]
        x = [xx for xx in x for _ in range(self.nImportanceSamples)]

        baseline_shared = self.baseline_shared()
        baseline_residual = self.baseline_residual()[i]
        baseline = baseline_shared + baseline_residual

        c, encscore = self.encoder(i, x)
        _, priorscore = self.prior(i, c)
        _, likelihood = self.decoder(i, c, x)
        kl = encscore - priorscore
        f = likelihood - (1-annealing)*kl
        f_reinforce = (f-baseline).data * encscore
        baselinescore = -(f-baseline)**2
        objective = f + f_reinforce + baselinescore
        elbo = f if annealing==0 else likelihood - kl
        objective = objective - objective.data + elbo.data

        if self.nImportanceSamples>1:
            objective = torch.logsumexp(
                    objective.reshape(batch_size, self.nImportanceSamples),
                    dim=1) - math.log(self.nImportanceSamples)
            kl = torch.mean(
                    kl.reshape(batch_size, self.nImportanceSamples),
                    dim=1)

        if return_kl:
            return objective, kl
        else:
            return objective

class VAE(IWAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, nImportanceSamples=1)


class RWS(EncoderDecoder):
    def __init__(self, encoder, decoder, prior, nTasks, nImportanceSamples=1, rObjective="sleep"):
        super().__init__()
        self.nTasks = nTasks
        self.nImportanceSamples = nImportanceSamples
        self.rObjective = rObjective

        self.t = Parameter(torch.zeros(1))

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def sample(self, i, x):
        cs = []
        weights = []
        for _ in range(self.nImportanceSamples):
            c, encscore = self.encoder(i, x)
            _, priorscore = self.prior(i, c)
            _, decscore = self.decoder(i, c, x)
            cs.append(c)
            weight = priorscore + decscore
            weights.append(weight.detach())

        weight=torch.stack(weights, dim=0)
        max_weight, max_idx = torch.max(weight, dim=0)
        
        #c = torch.stack(cs, dim=0)
        c = [cs[ii][jj] for jj,ii in enumerate(max_idx)]
        x, _ = self.decoder(i, c)
        return c, x

    def forward(self, i, x, return_kl=False, annealing=0, fixed=False):
        batch_size = len(i)

        #train r
        r_score = None
        if self.rObjective == "sleep" or self.rObjective == "mix":
            c, priorscore = self.prior(i)
            _x, likelihood = self.decoder(i, c)
            _, encscore = self.encoder(i, nograd(_x), c)
            r_score_sleep = encscore

        #wake
        i = [ii for ii in i for _ in range(self.nImportanceSamples)]
        x = [xx for xx in x for _ in range(self.nImportanceSamples)]

        c, encscore = self.encoder(i, x)
        _, priorscore = self.prior(i, c)
        _, likelihood = self.decoder(i, c, x)
        kl = encscore - priorscore
        p_score = priorscore + likelihood - encscore.data

        if self.nImportanceSamples > 1:
            log_w = p_score.reshape(batch_size, self.nImportanceSamples)
            lse = torch.logsumexp(log_w, dim=1)
            log_w_norm = log_w - lse[:, None]

            p_score = (log_w_norm.exp().data * log_w).sum(dim=1)
            p_score = p_score - p_score.data + (lse - math.log(self.nImportanceSamples))
            if self.rObjective == "wake" or self.rObjective == "mix":
                r_score_wake = (log_w_norm.exp().data * encscore.reshape(batch_size, self.nImportanceSamples)).sum(dim=1)
            kl = torch.mean(
                    kl.reshape(batch_size, self.nImportanceSamples),
                    dim=1)

        r_score = (r_score_wake if self.rObjective in ['wake', 'mix'] else 0) + \
                  (r_score_sleep if self.rObjective in ['sleep', 'mix'] else 0)

        objective = p_score + r_score - r_score.data
        if return_kl:
            return objective, kl
        else:
            return objective


    def conditional(self, i, D, x, n_samples=None, f=None):
        assert n_samples is None
        batch_size = len(i)

        i = [ii for ii in i for _ in range(self.nImportanceSamples)]
        D = [dd for dd in D for _ in range(self.nImportanceSamples)]
        x = [xx for xx in x for _ in range(self.nImportanceSamples)]

        if f is None: f = lambda c: self.decoder(i, c, x)[1]
        c, encscore = self.encoder(i, D)
        _, priorscore = self.prior(i, c)
        _, likelihood = self.decoder(i, c, D) 
        _weights = (priorscore + likelihood - encscore).reshape(batch_size, self.nImportanceSamples)
        log_weights = torch.log_softmax(_weights, dim=1)

        predictive = f(c).reshape(batch_size, self.nImportanceSamples)
        scores = torch.logsumexp(log_weights + predictive, dim=1) - math.log(self.nImportanceSamples)
       
        return scores

class WS(RWS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, nImportanceSamples=1)
