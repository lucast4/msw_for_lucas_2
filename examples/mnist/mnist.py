from builtins import super
import argparse
import math
import shutil
import os

from PIL import Image

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.distributions.bernoulli import Bernoulli
#from torch.distributions.categorical import Categorical
#from torch.distributions.beta import Beta
from torch.distributions.multivariate_normal import MultivariateNormal


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=['mnist', 'omniglot'], default='mnist')
#parser.add_argument('--data', type=str, choices=['mnist', 'omniglot'], default='mnist')
parser.add_argument('--nSupport', type=int, default=1)
parser.add_argument('--px', choices=['nade', 'mlp', 'cnn', 'pixelcnn', 'linear', 'positive', 'ink', 'gaussian', 'lines', 'stamps', 'curve','curve2', 'turtle','turtlemax','arc','arcmax'], default='arcmax')
parser.add_argument('--px_transform', choices=['same', 'different'], default='different')
parser.add_argument('--pc', choices=['nade', 'uniform', 'bernoulli', 'discrete'], default='nade')
parser.add_argument('--nComponents', type=int, default=256)
parser.add_argument('--rc', choices=['nade', 'cnn', 'linear'], default='nade')
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--nLatent', type=int, default=90)
parser.add_argument('--nSpatialLatent', type=int, default=0)
parser.add_argument('--nHidden', type=int, default=8)
args, unknown_args = parser.parse_known_args()

perm = torch.randperm(28*28)
nContentLatent = args.nLatent - args.nSpatialLatent
eyeperm = torch.eye(28*28)[:, perm[:nContentLatent]]
bperm = torch.zeros(28*28)
bperm[perm[:nContentLatent]]=1
cv=6
cb=-3
#Generate Data
np.random.seed(0)
if args.data=='mnist':
    nSupport = 1
    data_numpy = np.fromfile("./examples/mnist/data/binarized_mnist_train.amat", dtype=np.int16).reshape(-1,nSupport,28,28)
    np.random.shuffle(data_numpy)
    testdata_numpy = np.fromfile("./examples/mnist/data/binarized_mnist_test.amat", dtype=np.int16).reshape(-1,nSupport,28,28)
    #np.random.shuffle(testdata_numpy)
    data = (torch.from_numpy(data_numpy)-48).byte()
    data_smooth = data.float()
    testdata = (torch.from_numpy(testdata_numpy)-48).byte()

#elif args.data=='omniglot':
#    import scipy.io
#    mat = scipy.io.loadmat("examples/mnist/data/chardata_omniglot.mat")
#    np.random.seed(0)
#    d = mat['data'].transpose()
#    np.random.shuffle(d)
#    data = torch.Tensor((np.random.rand(*d.shape) < d).astype(np.uint8)).byte()
#    data = data.reshape(data.size(0), 28, 28)
#    d = mat['testdata'].transpose()
#    np.random.shuffle(d)
#    testdata = torch.Tensor((np.random.rand(*d.shape) < d).astype(np.uint8)).byte()
#    testdata = testdata.reshape(testdata.size(0), 28, 28)
#    args.nSupport=1
elif args.data=="omniglot":
    import scipy.io
    from collections import Counter
    mat = scipy.io.loadmat("examples/mnist/data/chardata_omniglot.mat")
    d = np.concatenate([
        mat['data'].transpose(),
        mat['testdata'].transpose()])
    labels = 100 * np.concatenate([
            mat['target'].transpose().nonzero()[1],
            mat['testtarget'].transpose().nonzero()[1]
        ]) + np.concatenate([
            mat['targetchar'][0],
            mat['testtargetchar'][0]
        ])
    np.random.seed(0)

    nRepeats = int(19/args.nSupport - 1)
    nTrain = nRepeats*args.nSupport
    nTest = args.nSupport

    alldata_train = np.concatenate([d[labels==k][None, :nTrain, :] for k,v in Counter(labels).items()]).reshape(-1, args.nSupport, 28, 28)
    alldata_test  = np.concatenate([d[labels==k][None, nTrain:nTrain+nTest, :] for k,v in Counter(labels).items()])\
            .reshape([-1, 1, args.nSupport, 28, 28])\
            .repeat(nRepeats, axis=1)\
            .reshape(alldata_train.shape)
    alldata = np.concatenate([alldata_train, alldata_test], axis=1)
    np.random.shuffle(alldata)
    #alldata_smooth = alldata
    alldata_smooth = torch.Tensor((0.5 < alldata).astype(np.uint8)).byte()
    alldata = torch.Tensor((np.random.rand(*alldata.shape) < alldata).astype(np.uint8)).byte()
    background_train = alldata[:, :args.nSupport]
    data = background_train
    data_smooth = alldata_smooth[:, :args.nSupport]

    background_test = alldata[:, args.nSupport:2*args.nSupport]
    testdata = background_test
    #background_train = alldata[:-300, :args.nSupport]
    #background_test = alldata[:-300, args.nSupport:2*args.nSupport]
    #novel_train = alldata[-300:, :args.nSupport]
    #novel_test = alldata[-300:, args.nSupport:2*args.nSupport]

if args.n is not None:
    data = data[:args.n]
testdata = testdata[:2000]
print("got %d instances" % len(data))

#data = data.cuda() #LT
#testdata = testdata.cuda() #LT

def t(x, tens): # converts to tensor.
    if torch.is_tensor(x):
        return x
    elif torch.is_tensor(x[0]):
        return torch.stack(x)
    else:
        return tens.new_tensor(x)

class NADE(nn.Module):
    def __init__(self, length, nHidden=None, nConditional=None, lr=1):
        super().__init__()
        self.length = length
        if nHidden is None: nHidden=length
        self.nHidden = nHidden
        self.lr = 1

        self.t = Parameter(torch.ones(1))
        self.triu = Parameter(torch.ones(length, length).triu(diagonal=1)) #length_in x length_out

        r=1

        self._W = Parameter(torch.randn(length,nHidden)*math.sqrt(2./length)*r * self.lr)
        self._a = Parameter(torch.zeros(nHidden) * self.lr)

        self._V = Parameter(torch.randn(length,nHidden)*math.sqrt(2./nHidden)*r * self.lr)
        self._b = Parameter(torch.zeros(length) * self.lr)

        self.conditional = nConditional is not None
        if self.conditional:
            self._Ua = Parameter(torch.randn(nConditional, nHidden)*math.sqrt(2./nConditional)*r * self.lr)
            self._Ub = Parameter(torch.randn(nConditional, length)*math.sqrt(2./length)*self.lr)

    @property
    def W(self): return self._W / self.lr

    @property
    def a(self): return self._a / self.lr

    @property
    def V(self): return self._V / self.lr

    @property
    def b(self): return self._b / self.lr

    @property
    def Ua(self): return self._Ua / self.lr

    @property
    def Ub(self): return self._Ub / self.lr

    def logit_p_elementwise(self, x, y=None):
        assert self.conditional == (y is not None)
        # x: batch * length
        # y: batch * nConditional
        if self.conditional:
            h = torch.tanh(
                    torch.einsum('bi,ih,ij->bjh', (x, self.W, self.triu.data))  # batch * length_out * nHidden
                    + (self.Ua[None, None, :, :] * y[:, None, :, None]).sum(dim=2)
                    + self.a[None, None, :])
            logit_p = torch.einsum('bjh,jh->bj', (h, self.V)) \
                    + (self.Ub[None, :, :] * y[:, :, None]).sum(dim=1) 
        else:
            h = torch.tanh(
                    torch.einsum('bi,ih,ij->bjh', (x, self.W, self.triu.data))  # batch * length_out * nHidden
                    + self.a[None, None, :])
            logit_p = torch.einsum('bjh,jh->bj', (h, self.V))
        return logit_p

    def score(self, x, y=None):
        # x: batch * length
        logit_p = self.logit_p_elementwise(x, y=y)
        scores = -nn.functional.binary_cross_entropy_with_logits(logit_p, x, reduction='none')
        return scores.sum(dim=1)

    def sampleAndScore(self, batch_size=None, y=None):
        if self.conditional:
            assert batch_size is None and y is not None
            batch_size = y.size(0)
        else: assert batch_size is not None and y is None

        xs = []
        hs = []
        #x = self.t.new_zeros(batch_size, self.length)

        if self.conditional:
            h_inp = self.a.data[None, :].repeat(batch_size, 1) \
                + (self.Ua[None, :, :] * y[:, :, None]).sum(dim=1)
        else:
            h_inp = self.a.data[None, :].repeat(batch_size, 1).clone()

        score = self.t.new_zeros(batch_size)
        
        for i in range(self.length):
            h = torch.tanh(h_inp) # batch * nHidden
            if self.conditional:
                p = torch.sigmoid(
                        (h * self.V.data[None,i,:]).sum(dim=1)
                        + (self.Ub[None, :, i] * y[:, :]).sum(dim=1)
                        + self.b.data[None,i]) #batch
            else:
                p = torch.sigmoid(
                        (h * self.V.data[None,i,:]).sum(dim=1)
                        + self.b.data[None,i]) #batch

            #x[:, i] = Bernoulli(p).sample()
            #score = score + Bernoulli(p).log_prob(x[:, i])
            #h_inp = h_inp + x[:, i, None] * self.W.data[None, i, :]
            xs.append(Bernoulli(p).sample())
            hs.append(h)
            score = score + Bernoulli(p).log_prob(xs[i])
            h_inp = h_inp + xs[i][:, None] * self.W.data[None, i, :]

        x = torch.stack(xs, 1)
        hs = torch.stack(hs,1)
        
        return x, score

class RcTransform(nn.Module):
    def __init__(self, rc, transform):
        super().__init__()
        self.rc = rc
        self.transform = transform

    def forward(self, i, x, c=None, subsample=True):
        return self.rc(i, self.transform(x), c, subsample)

class PxIID(nn.Module):
    def __init__(self, px, n):
        super().__init__()
        self.px = px
        self.n = n

    def forward(self, i, c, x=None, *args, **kwargs):
        if x is None:
            _x = None
        elif torch.is_tensor(x):
            #_x = x[:, None].expand(x.size(0), self.n, *x.size()[1:]).reshape(x.size(0)*self.n, x.size()[1:])
            #raise NotImplementedError()
            _x = [xxi for xx in x for xxi in xx]
        else:
            _x = [xxi for xx in x for xxi in xx]

        #_i = i[:, None].expand(-1, self.n).reshape(-1)
        if i is None:
            _i = None
        else:
            _i = [ii for ii in i for _ in range(self.n)]
        _c = [cc for cc in c for _ in range(self.n)]
        _x, _score = self.px(_i, _c, _x, *args, **kwargs)

        if torch.is_tensor(_x):
            x = _x.reshape(-1, self.n, 28, 28)
        else:
            x = list(torch.cat(_x).reshape(-1, self.n, 28, 28))
        score = _score.reshape(len(c), -1).sum(dim=1)
        return x, score


class PcNade(nn.Module):
    def __init__(self, nLatent=args.nLatent):
        super().__init__()
        if args.nLatent!=0:
            self.nade = NADE(args.nLatent, nHidden=args.nHidden, lr=1)
            self.modules = nn.ModuleList([self.nade])
        self.t = Parameter(torch.ones(1))

    def forward(self, i, c=None):
        if args.nLatent==0: return [None]*len(i),self.t.new_zeros(len(i))
        if c is None:
            c, score = self.nade.sampleAndScore(batch_size=len(i))
            c = c.tolist()
        else:
            score = self.nade.score(t(c, self.t))
        return c, score 

    #def hyperprior(self):      # This is waaay unstable, obviously
    #    n_samples = 100
    #    c = (self.t.new_tensor(torch.rand(n_samples, args.nLatent))>0.5).float()
    #    p = self.nade.logit_p_elementwise(c).sigmoid()
    #    one = self.t.data
    #    dist = Beta(one, one+1)
    #    hyperprior_estimate = (dist.log_prob(p) * 2**self.t.new_tensor(range(args.nLatent))[None]).sum()/n_samples
    #    return hyperprior_estimate

class PcBernoulli(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = Parameter(torch.ones(1))
        if args.nLatent != 0:
            self.logits = Parameter(torch.zeros(args.nLatent))

    def forward(self, i, c=None):
        if args.nLatent==0: return [None]*len(i),self.t.new_zeros(len(i))
        dist = Bernoulli(logits=self.logits[None, :])
        if c is None:
            c_tens = dist.sample(sample_shape=(len(i),))[:,0,:]
            c = c_tens.tolist()
        else:
            c_tens = self.t.new_tensor(c)
        score = dist.log_prob(c_tens).sum(dim=1) 
        return c, score 

    #def hyperprior(self):
    #    p = self.logits.sigmoid()
    #    one = self.t.data
    #    dist = Beta(one, one+2)
    #    #return ((p-0.5).abs()*4).log().sum()
    #    #return (2*(1-p)).log().sum()
    #    return dist.log_prob(p).sum()

class PcDiscrete(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = Parameter(torch.ones(1))
        if args.nLatent != 0:
            self.logits = Parameter(torch.zeros(args.nary))

    def forward(self, i, c=None):
        assert args.nLatent != 0
        dist = Bernoulli(logits=self.logits[None, :])
        if c is None:
            c_tens = dist.sample(sample_shape=(len(i)*args.nLatent,))
            c_tens = c_tens.reshape(len(i), args.nLatent)
            c = c_tens.tolist()
        else:
            c_tens = self.t.new_tensor(c)
        score = dist.log_prob(c_tens).sum(dim=1) 
        return c, score 

class PcUniform(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = Parameter(torch.ones(1))

    def forward(self, i, c=None):
        if args.nLatent==0: return [None]*len(i),self.t.new_zeros(len(i))
        if c is None:
            c = (torch.rand(len(i), args.nLatent)>0.5).tolist()
        score = self.t.new_full((len(i),), math.log(2)*args.nLatent)
        return c, score 

if args.pc=="nade": Pc = PcNade
elif args.pc=="uniform": Pc=PcUniform
elif args.pc=="bernoulli": Pc=PcBernoulli
elif args.pc=="discrete": Pc=PcDiscrete
#elif args.pc="bernade":
#    class Pc(nn.Module):
#        def __init__(self):
#            super().__init__()
#            self.nade = NADE(args.nSpatialLatent, nHidden=args.nHidden)
#            self.t = Parameter(torch.ones(1))
#            self.logits = Parameter(torch.zeros(nContentLatent))
#
#    def forward(self, i, c=None):
#        if args.nLatent==0: return [None]*len(i),self.t.new_zeros(len(i))
#
#        dist_bern = Bernoulli(logits=self.logits[None, :])
#        if c is None:
#            c_tens_bern = dist.sample(sample_shape=(len(i),))[:,0,:]
#            score_bern = dist.log_prob(c_tens_bern).sum(dim=1) 
#            c_nade, score_nade = self.nade.sampleAndScore(batch_size=len(i))
#            c = torch.cat([c_bern, c_nade], dim=1).tolist()
#        else:
#            c_tens_bern = self.t.new_tensor([c[:nContentLatent] for cc in c])
#            score_bern = dist.log_prob(c_tens_bern).sum(dim=1) 
#            score_nade = self.nade.score(t(c, self.t)[nContentLatent:])
#        score = score_bern + score_nade
#
#    def hyperprior(self):
#        p = self.logits.sigmoid()
#        one = self.t.data
#        dist = Beta(one, one+2)
#        return dist.log_prob(p).sum()
#


class MLP(nn.Module):
    def __init__(self, dIn, nHidden=None):
        super().__init__()
        if nHidden is None: nHidden = dIn
        self.fc1 = nn.Linear(dIn, nHidden)
        self.fc2 = nn.Linear(nHidden, 28*28)

    def forward(self, x, extra=None):
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        
        return y.view(-1, 28, 28)

class CNNp(nn.Module):
    # THIS IS IN A BROKEN STATE
    def __init__(self, dIn, nHidden=500):
        super().__init__()
        self.fc1=nn.Linear(dIn, nHidden)
        self.fc2 = nn.Linear(nHidden, 4*4*50)
        self.conv1 = nn.ConvTranspose2d(50, 20, 5, 2)
        self.conv2 = nn.ConvTranspose2d(20, 20, 4, 2)
        self.conv3 = nn.ConvTranspose2d(20, 1, 5, 1)

        ct=nn.ConvTranspose2d
        self.conv1=ct(50,50,3) #6x6
        self.conv2=ct(50,50,3,stride=2) #13x13
        #self.conv3=ct(50,1,4,stride=2)

        #self.foo_
        self.foo = nn.Linear(50*13*13, 28*28)
        #for w in [self.fc1.weight, self.fc2.weight, self.conv1.bias, self.conv1.weight, self.conv2.bias, self.conv2.weight, self.conv3.bias, self.conv3.weight]:
        #    nn.init.normal_(w, std=0.1)

    def forward(self, x, extra=None):
        y = F.sigmoid(self.fc1(x))
        y = F.sigmoid(self.fc2(y))
        y = y.view(-1, 50, 4, 4)
        y = F.sigmoid(self.conv1(y))
        y = F.sigmoid(self.conv2(y))
        out=self.foo(y.view(-1, 50*13*13))
        #out = self.conv3(y)

        
        return out.view(-1, 28, 28)

class CNN(nn.Module):
    def __init__(self, dOut, nExtraInput=0):
        super().__init__()
        self.nExtraInput = nExtraInput
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50 + nExtraInput, 500)
        self.fc2 = nn.Linear(500, dOut)

    def forward(self, x, extra=None):
        y = F.relu(self.conv1(x))
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2, 2)
        y = y.view(-1, 4*4*50)
        if extra is not None:
            y = torch.cat([y, extra], dim=1)
        y = self.fc1(y)
        y = y.view(len(x), args.nSupport, y.size(-1)).mean(dim=1)
        y = F.relu(y)
        out = self.fc2(y)
        return out

class SpatialCNN(CNN):
    def __init__(self):
        super().__init__(6 + args.nSpatialLatent)
        self.spatial_lr=0.01
        self.fc2.bias.data[:6].copy_(torch.eye(2,3).reshape(6)/self.spatial_lr)
        self.fc2.weight.data.zero_()
        self.t = Parameter(torch.ones(1))

    def forward(self, x):
        out = super().forward(x)
        theta = out[:, :6].reshape(-1, 2, 3)*self.spatial_lr
        spatialLatent = out[:, 6:]
        return theta, spatialLatent

if args.rc == "nade":
    class Rc(nn.Module):
        def __init__(self):
            super().__init__()
            if args.nLatent!=0:
                self.use_average=True
                self.nade = NADE(args.nLatent, nHidden=args.nHidden, nConditional=28*28 if self.use_average else args.nSupport*28*28)
                self.modules = nn.ModuleList([self.nade])
            self.t = Parameter(torch.ones(1))

        def forward(self, i, x, c=None, subsample=True):
            if args.nLatent==0: return [None]*len(i), self.t.new_zeros(len(i))
            if self.use_average:
                x_flat = t(x, self.t).reshape(len(x), args.nSupport, 28*28).float().mean(dim=1)
            else:
                x_flat = t(x, self.t).reshape(len(x), args.nSupport*28*28)

            if c is None:
                c, score = self.nade.sampleAndScore(y=x_flat.float())
                c = c.tolist()
            else:
                score = self.nade.score(t(c, self.t), y=x_flat.float())
            return c, score 
elif args.rc=="cnn":
    class Rc(nn.Module):
        def __init__(self):
            super().__init__()
            if args.nLatent!=0:
                self.spatialCNN = SpatialCNN()
                self.conv = CNN(nContentLatent)
            self.t = Parameter(torch.zeros(1))

        def forward(self, i, x, c=None, subsample=True):
            if args.nLatent==0: return [None]*len(i), self.t.new_zeros(len(i))
            x_inp = t(x, self.t).reshape(len(x)*args.nSupport, 1, 28, 28).float()
            theta, spatialLatent = self.spatialCNN(x_inp)
            x_trans = F.grid_sample(x_inp, F.affine_grid(theta, x_inp.size()))
            logits = torch.cat([self.conv(x_trans), spatialLatent], dim=1)#, extra=theta.reshape(-1, 6))

            dist = Bernoulli(logits=logits)
            if c is None:
                c_tens = dist.sample()
                c = c_tens.tolist()
            else:
                c_tens=self.t.new_tensor(c)
            score = dist.log_prob(c_tens).sum(dim=1)
            return c, score 

elif args.rc=="linear":
    class Rc(nn.Module):
        def __init__(self):
            super().__init__()
            assert args.nLatent>0
            #assert args.nLatent>=28*28
            if args.nLatent<=28*28:
                self.v = Parameter(eyeperm.clone()*cv)
            else:
                self.v = Parameter(torch.eye(28*28, nContentLatent) * cv)
            self.b = Parameter(torch.ones(nContentLatent)*cb)
            self.spatialCNN = SpatialCNN()
            self.t = Parameter(torch.zeros(1))

        def forward(self, i, x, c=None, subsample=True):
            x_inp = t(x, self.t).reshape(len(x),args.nSupport, 1, 28, 28).float().mean(dim=1)
            theta, spatialLatent  = self.spatialCNN(x_inp)
            x_trans = F.grid_sample(x_inp, F.affine_grid(theta, x_inp.size()))
            #logits = self.b[None, :] + x_inp.reshape(-1, 28*28).mm(self.v)
            logits = self.b[None, :] + x_trans.reshape(-1, 28*28).mm(self.v)
            logits = torch.cat([logits, spatialLatent], dim=1)#+ self.getTransLatent(theta.reshape(-1, 6))
            dist = Bernoulli(logits=logits)
            if c is None:
                c_tens = dist.sample()
                c = c_tens.tolist()
            else:
                c_tens=self.t.new_tensor(c)
            score = dist.log_prob(c_tens).sum(dim=1)
            return c, score 



if args.px=="nade":
    class _Px(nn.Module):
        def __init__(self):
            super().__init__()
            if args.nLatent==0: self.nade = NADE(28*28, nHidden=args.nHidden)
            else: self.nade = NADE(28*28, nHidden=args.nHidden, nConditional=args.nLatent)
            self.modules = nn.ModuleList([self.nade])
            self.t = Parameter(torch.ones(1))

        def forward(self, i, c, x=None):
            if x is None:
                if args.nLatent==0: x_flat, score = self.nade.sampleAndScore(batch_size=len(i))
                else: x_flat, score = self.nade.sampleAndScore(y=t(c, self.t))
                x = x_flat.reshape(x_flat.size(0), 28, 28).byte()
            else:
                x_flat = t(x, self.t).reshape(len(x), 28*28)
                if args.nLatent==0: score = self.nade.score(x_flat.float())
                else: score = self.nade.score(x_flat.float(), y=t(c, self.t))
            return x, score 

elif args.px in ["turtle", 'turtlemax', 'arc', 'arcmax']:
    class _Px(nn.Module):
        def __init__(self):
            super().__init__()
            if args.nComponents == 8: self.binary = Parameter(torch.Tensor([4,2,1]))
            elif args.nComponents == 16: self.binary = Parameter(torch.Tensor([8,4,2,1]))
            elif args.nComponents == 32: self.binary = Parameter(torch.Tensor([16,8,4,2,1]))
            elif args.nComponents == 64: self.binary = Parameter(torch.Tensor([32,16,8,4,2,1]))
            elif args.nComponents == 128: self.binary = Parameter(torch.Tensor([64,32,16,8,4,2,1]))
            elif args.nComponents == 256: self.binary = Parameter(torch.Tensor([128,64,32,16,8,4,2,1]))
            elif args.nComponents == 512: self.binary = Parameter(torch.Tensor([256,128,64,32,16,8,4,2,1]))
            elif args.nComponents == 1024: self.binary = Parameter(torch.Tensor([512,256,128,64,32,16,8,4,2,1]))
            else:
                print(args.nComponents)
                raise NotImplementedError()

            assert(nContentLatent % (1+len(self.binary)) == 0)
            self.nStrokes = int(nContentLatent/(1+len(self.binary)))

            self._v_scale = Parameter(torch.ones(1) * 3)
            self._b = Parameter(torch.ones(1)*-3)

            self._x = Parameter((torch.arange(28)[None, :].float().expand(28, 28)+0.5)/28)
            self._y = Parameter(((27-torch.arange(28))[:, None].float().expand(28, 28)+0.5)/28)
            self.angle = Parameter(torch.randn(args.nComponents, 1, 1)*math.pi*0.25)#*2*math.pi)

            self._w1 = Parameter(torch.ones(args.nComponents) * 0 - 2 )
            self._w2 = Parameter(torch.ones(args.nComponents) * 0 - 2)
            self._u1 = Parameter(torch.ones(args.nComponents) * 20 )
            self._u2 = Parameter(torch.ones(args.nComponents) * 20)
            self._dx = Parameter(torch.randn(args.nComponents, 1, 1)*0.2) 
            self._dy = Parameter(torch.randn(args.nComponents, 1, 1)*0.2) 

            self._top = Parameter(torch.Tensor([0]))

            if args.nSpatialLatent>0:
                self.spatial = nn.Linear(args.nSpatialLatent, 6)
                self.spatial_lr=0.01
                self.spatial.bias.data.copy_(torch.eye(2,3).reshape(6)/self.spatial_lr)
                self.spatial.weight.data.zero_()
            self.t = Parameter(torch.ones(1))
            self.truncate_to=None

        @property
        def dx(self): return self._dx.tanh()*0.66 #<-------------------- THIS IS THE MAXIMUM IT CAN JUMP
        @property
        def dy(self): return self._dy.tanh()*0.66
        @property
        def top(self): return self._top #F.softplus(self._top)
        
        @property
        def w1(self): return self._w1.sigmoid()*0.1 + 0 #<-------------- LINE WIDTH
        #def w1(self): return F.softplus(self._w1)
        
        @property
        def w2(self): return self._w2.sigmoid()*0.1 + 0
        #def w2(self): return F.softplus(self._w2)

        @property
        def u1(self): return self._u1
        
        @property
        def u2(self): return self._u2

        def l2(self, w, h):
            return h**2 + w**2

        def get_theta(self, c_spatial):
            theta = self.spatial(c_spatial).view(-1, 2, 3)*self.spatial_lr
            return theta

        def get_lines(self, c_idx=None, x_offset=None, y_offset=None, noise=0):
            w1 = self.w1 if c_idx is None else self.w1[c_idx]
            w2 = self.w2 if c_idx is None else self.w2[c_idx]
            u1 = self.u1 if c_idx is None else self.u1[c_idx]
            u2 = self.u2 if c_idx is None else self.u2[c_idx]
            dx = self.dx if c_idx is None else self.dx[c_idx]
            dy = self.dy if c_idx is None else self.dy[c_idx]
            angle = self.angle if c_idx is None else self.angle[c_idx]
            _x = self._x.data[None] if x_offset is None else self._x.data[None] - x_offset[:, None, None]
            _y = self._y.data[None] if y_offset is None else self._y.data[None] - y_offset[:, None, None]

            if noise > 0:
                dx *= 1 + dx.new(dx.size()).normal_()*noise
                dy *= 1 + dy.new(dx.size()).normal_()*noise

            if args.px in ["arc", 'arcmax']:
               d = (dx**2 + dy**2).sqrt()

               #perpendicular distance to center of circle
               l = (d/2) / torch.tan(angle)
               
               # center of circle
               x0 = dx*0.5 + l * dy/d
               y0 = dy*0.5 - l * dx/d
               r = (x0**2 + y0**2).sqrt()

               # vector from center
               dx0 = _x - x0
               dy0 = _y - y0
               _r = (dx0**2 + dy0**2).sqrt()

               #arc
               clockwise = (angle%(2*math.pi))<math.pi
               _angle = torch.atan2(_x-x0, _y-y0)
               angle1 = torch.atan2(-x0, -y0)
               angle2 = torch.atan2(dx-x0, dy-y0)
               _angle_delta = (_angle-angle1)%(2*math.pi) 
               angle2_delta = (angle2-angle1)%(2*math.pi) 
               in_clockwise = (_angle_delta < angle2_delta)
               in_arc = (in_clockwise == clockwise).float()

               sqdist_circle = (r - _r)**2
               sqdist_1 = _x**2 + _y**2
               sqdist_2 = (_x-dx)**2 + (_y-dy)**2
               sqdist_endpoints = torch.stack([sqdist_1, sqdist_2]).min(dim=0)[0]

               sqdist = in_arc*sqdist_circle + (1-in_arc)*sqdist_endpoints

               _t_clockwise = _angle_delta/angle2_delta
               _t_clockwise[1-in_clockwise]=0
               _t_counter   = (2*math.pi-_angle_delta)/(2*math.pi-angle2_delta)
               _t_counter[in_clockwise]=0
               _t = _t_clockwise + _t_counter
               #_t = in_clockwise*_angle_delta/angle2_delta + (1-in_clockwise)*(2*math.pi-_angle_delta)/(2*math.pi-angle2_delta)


            elif args.px in ['turtle', 'turtlemax']:
                #x0 = dx*0 + 0.5
                x1 = dx*0 + 0.5
                #y0 = dy*0 + 0.5
                y1 = dx*0 + 0.5

                dx0 = self._x.data - x1
                dy0 = self._y.data - y1
                _t = ((dx0*dx + dy0*dy) / self.l2(dx,dy)).clamp(0,1)

                sqdist = self.l2(dx0 - _t*dx, dy0 - _t*dy)

            w = (1-_t)*w1[:, None, None] + _t*w2[:, None, None]
            u = (1-_t)*u1[:, None, None] + _t*u2[:, None, None]
            lines = (u**2 * (w**2-sqdist)).exp() 
            lines = torch.stack([lines, self.t.new_ones(1).expand_as(lines)]).min(dim=0)[0] #From 0 to 1
            if torch.isnan(lines).sum()>0:
                import pdb
                pdb.set_trace()
           
            return lines

        def get_all_strokes(self):
            lines = self.get_lines()
            logits = lines.view(-1, 28, 28)
            
            logits = F.softplus(self._v_scale) * logits 
            logits = logits + self._b

            p = logits.sigmoid() 
            return p

        def forward(self, i, c, x=None, sample_probs=False, noise=0):
            c_tens = self.t.new_tensor(c)#.reshape(len(c), args.nLatent)
            c_content = c_tens[:, :nContentLatent]
            c_on = c_content[:, :self.nStrokes]
            c_lines = c_content[:, self.nStrokes:]
            if args.nSpatialLatent>0:
                c_spatial = c_tens[:, nContentLatent:]
            c_idx = (c_lines.view(len(c_tens), self.nStrokes, len(self.binary)) * self.binary.data[None, None, :]).sum(dim=2).long()
            z = self.t.new_zeros(len(c_idx), 1, 1, 1)
            dx = self.dx[c_idx]
            dy = self.dy[c_idx]
            x1 = torch.cat([z, dx[:, :-1].cumsum(dim=1)], dim=1) + 0.5
            y1 = torch.cat([z, dy[:, :-1].cumsum(dim=1)], dim=1) + 0.5

            #w1 = self.w1[c_idx]
            #w2 = self.w2[c_idx]
            #u1 = self.u1[c_idx]
            #u2 = self.u2[c_idx]
            #

            #dx0 = self._x.data - x1
            #dy0 = self._y.data - y1
            #_t = ((dx0*dx + dy0*dy) / self.l2(dx,dy)).clamp(0,1)
            #w = (1-_t)*w1[:, :, None, None] + _t*w2[:, :,None, None]
            #u = (1-_t)*u1[:, :, None, None] + _t*u2[:, :,None, None]
            #sqdist = self.l2(dx0 - _t*dx, dy0 - _t*dy)
            #lines = (u**2 * (w**2-sqdist)).exp() 
            #lines = torch.stack([lines, self.t.new_ones(1).expand_as(lines)]).min(dim=0)[0] #From 0 to 1
            lines = self.get_lines(c_idx.view(-1), x1.view(-1), y1.view(-1), noise=noise)

            logits = lines.view(-1, self.nStrokes, 28, 28)
            logits = logits*c_on[:, :, None, None]
            logits = logits[:, 1:] #<------------------------- First line is always ignored
            if self.truncate_to is not None:
                logits = logits[:, :self.truncate_to]
            
            if args.px=="turtle" or args.px=="arc":
                logits = logits.sum(dim=1)
            elif args.px=="turtlemax" or args.px=="arcmax":
                logits = logits.max(dim=1)[0] # max scaling over all strokes images (i.e final image)
            logits = F.softplus(self._v_scale) * logits 

            if args.nSpatialLatent>0:
                theta = self.get_theta(c_spatial)
                logits = logits[:, None, :, :]
                logits = F.grid_sample(logits, F.affine_grid(theta, logits.size()))[:, 0]
            logits = logits + self._b

            p = logits.sigmoid() 

            dist = Bernoulli(logits = logits)
            if x is None:
                x_float = dist.sample() 
                x_byte = x_float.byte()
            else:
                x_byte = t(x, self.t).byte()
                x_float = x_byte.float()
            score = dist.log_prob(x_float).sum(dim=2).sum(dim=1)
            if sample_probs: return p, score
            else: return x_byte, score 



elif args.px in ["linear", "positive", "ink", "gaussian", "lines", "stamps", "curve", "curve2"]:
    class _Px(nn.Module):
        def __init__(self):
            super().__init__()
            assert args.nLatent>0
            if args.rc=="linear":
                if nContentLatent<=28*28:
                    self._v = Parameter(eyeperm.clone().t() * cv)
                    #self.b = Parameter(bperm.clone()*cb)
                else:
                    self._v = Parameter(torch.eye(nContentLatent, 28*28) * cv)
                    #self.b = Parameter(torch.ones(28*28)*cb)
            elif args.px=="ink":
                self._v = Parameter(torch.randn(nContentLatent, 28*28)*0.1 - 4)
            else:
                print("Linear pc without linear rc")
                self._v = Parameter(torch.zeros(nContentLatent, 28*28))
                #self.b = Parameter(torch.zeros(28*28))

            if args.px=="ink":
                self.b_single = Parameter(torch.zeros(1)-2)
            else:
                self.b_single = Parameter(torch.ones(1)*-3)
            
            self._v_lr = 0.001
            #for gaussian
            self._v_mu = Parameter(torch.rand(nContentLatent, 2) / self._v_lr)
            self._v_tri = Parameter(torch.eye(2)[None].expand(nContentLatent, 2, 2)*0.1 / self._v_lr)
            self._v_scale = Parameter(torch.zeros(nContentLatent))
            self._x = Parameter((torch.arange(28)[:, None].float().expand(28, 28)+0.5)/28)
            self._y = Parameter((torch.arange(28)[None, :].float().expand(28, 28)+0.5)/28)
            self.coords = Parameter(torch.stack([self._x.data,self._y.data], dim=2))
           
            #for lines
            self._v_template = torch.ones(28,28)*-3
            self._v_template[14:15, 5:22]=3
            self._v_template = Parameter(self._v_template)
            th = torch.rand(nContentLatent)*2*math.pi
            self._v_theta = Parameter(torch.stack([
                    torch.stack([th.cos(), th.sin()], dim=1),
                    torch.stack([-th.sin(), th.cos()], dim=1),
                    torch.randn(len(th), 2)*0.3
                ], dim=2) / self._v_lr)
            flip_idxs = torch.rand(len(th))>0.5
            self._v_theta.data[flip_idxs, :2, :2] = self._v_theta.data[flip_idxs, :2, :2].transpose(1,2)
            self._v_theta.data[:, :2, :2] = self._v_theta.data[:, :2, :2]/torch.rand(len(th), 1,1)

            #for curves
            self._w1 = Parameter(torch.randn(nContentLatent) - 7)
            self._w2 = Parameter(torch.randn(nContentLatent) - 7)
            self._w3 = Parameter(torch.randn(nContentLatent) - 7) #curve2
            self._x1 = Parameter(torch.randn(nContentLatent, 1, 1)) 
            self._y1 = Parameter(torch.randn(nContentLatent, 1, 1)) 
            self._dx = Parameter(torch.randn(nContentLatent, 1, 1)*0.1) 
            self._dy = Parameter(torch.randn(nContentLatent, 1, 1)*0.1) 
            self._ddx = Parameter(torch.randn(nContentLatent, 1, 1)*0.1) #curve2
            self._ddy = Parameter(torch.randn(nContentLatent, 1, 1)*0.1) #curve2


            if args.px_transform=="same":
                self.spatial = nn.Linear(args.nSpatialLatent, 6)
                self.spatial_lr=0.01
                self.spatial.bias.data.copy_(torch.eye(2,3).reshape(6)/self.spatial_lr)
                self.spatial.weight.data.zero_()
            elif args.px_transform=="different":
                self.spatial = nn.Linear(args.nSpatialLatent, 6*(nContentLatent))
                self.spatial_lr=0.01
                self.spatial.bias.data.copy_(torch.eye(2,3).reshape(6).repeat(nContentLatent)/self.spatial_lr)
                self.spatial.weight.data.zero_()
            self.t = Parameter(torch.ones(1))
        
        def l2(self, w, h):
            return h**2 + w**2


        @property
        def b(self):
            if args.px in ['ink']:
                return F.softplus(self.b_single).expand(28*28)
            else:
                return self.b_single.expand(28*28)

        @property
        def x(self): return self._x.data
        @property
        def y(self): return self._y.data
        @property
        def x1(self): return self._x1.sigmoid()
        @property
        def y1(self): return self._y1.sigmoid()
        @property
        def dx(self): return self._dx.tanh()
        @property
        def dy(self): return self._dy.tanh()
        @property
        def ddx(self): return self._ddx.tanh()
        @property
        def ddy(self): return self._ddy.tanh()
        @property
        def w1(self): return F.softplus(self._w1)
        @property
        def w3(self): return F.softplus(self._w3)
        @property
        def w2(self): return (self.w1 + self.w3)/2
        #def w2(self): return F.softplus(self._w2)

        def get_v(self, theta=None):
            if args.px=="gaussian":
                assert(theta is None)
                dist = MultivariateNormal(loc=self._v_mu[:,None,None,:]*self._v_lr, scale_tril=self._v_tri[:,None,None,:]*self._v_lr)
                score = dist.log_prob(self.coords.data[None]).exp()
                out = score/score.max() * F.softplus(self._v_scale[:, None, None])
                return out.view(-1, 28*28)
            elif args.px in ["lines", "stamps"]:
                assert(theta is None)
                v_template = F.softplus(self._v_template[None, None,:,:])
                if args.px=="lines": v_template=v_template.data
                v_template = v_template * F.softplus(self._v_scale[:, None, None, None])
                grid = F.affine_grid(self._v_theta*self._v_lr, v_template.size())
                lines = F.grid_sample(v_template, grid)
                return lines.view(nContentLatent, 28*28)
            elif "curve" in args.px:
                assert(theta is None)
                x=self.x
                y=self.y
                x1=self.x1
                y1=self.y1
                dx=self.dx
                dy=self.dy
                w1=self.w1
                w2=self.w2
                
                dx0 = x - x1
                dy0 = y - y1
                t = ((dx0*dx + dy0*dy) / self.l2(dx,dy)).clamp(0,1)
                w = (1-t)*w1[:, None, None] + t*w2[:, None, None]
                sqdist = self.l2(dx0 - t*dx, dy0 - t*dy)
                lines = (-sqdist/w).exp()
                if args.px=="curve2":
                    x2=x1+dx
                    y2=y1+dy
                    dx2 = self.ddx + dx
                    dy2 = self.ddy + dy
                    w3=self.w3

                    dx02=x-x2
                    dy02=y-y2
                    t2 = ((dx02*dx2 + dy02*dy2) / self.l2(dx2,dy2)).clamp(0,1)
                    w2 = (1-t2)*w2[:, None, None] + t2*w3[:, None, None]
                    sqdist2 = self.l2(dx02 - t2*dx2, dy02 - t2*dy2)
                    lines2 = (-sqdist2/w2).exp()
                    lines, _ = torch.stack([lines, lines2], dim=0).max(dim=0)

                lines = F.softplus(self._v_scale[:, None, None]) * lines
                return lines.view(nContentLatent, 28*28)
            elif args.px in ["positive", "ink"]:
                if theta is None:
                    return F.softplus(self._v)
                else:
                    # theta is batch*latents*2*3
                    v = self._v.repeat(theta.size(0), 1)
                    v = v.view(v.size(0), 1, 28, 28)
                    transformed_v = F.grid_sample(v, F.affine_grid(theta.view(-1, 2, 3), v.size())).view(theta.size(0), theta.size(1), 28*28)
                    return F.softplus(transformed_v)

            else:
                assert(theta is None)
                return self._v

        @property
        def v(self):
            return self.get_v()

        def get_theta(self, c_spatial):
            if args.px_transform=="same":
                theta = self.spatial(c_spatial).view(-1, 2, 3)*self.spatial_lr
            elif args.px_transform=="different":
                theta = self.spatial(c_spatial).view(-1, nContentLatent, 2, 3)*self.spatial_lr
            return theta

        def forward(self, i, c, x=None):
            c_tens = self.t.new_tensor(c)#.reshape(len(c), args.nLatent)
            c_content = c_tens[:, :nContentLatent]
            c_spatial = c_tens[:, nContentLatent:]
            theta = self.get_theta(c_spatial)
            
            if args.px_transform=="same":
                v=self.v
                # v is batch*(28*28)
                y = c_content.mm(v).view(len(c), 1, 28, 28)
                logits = F.grid_sample(y, F.affine_grid(theta, y.size()))[:, 0]
            elif args.px_transform=="different":
                # v is batch*latents*(28*28)
                v = self.get_v(theta)
                y = (c_content[:, :, None] * v).sum(dim=1)
                logits = y.view(len(c), 28, 28)

            logits = logits + self.b.reshape(1,28,28)
            
            if args.px == "ink":
                p = (logits.sigmoid()-0.5)*2
                dist = Bernoulli(probs = p)
            else:
                p = logits.sigmoid() 
                dist = Bernoulli(logits = logits)
            if x is None:
                x_float = dist.sample() 
                x_byte = x_float.byte()
            else:
                x_byte = t(x, self.t).byte()
                x_float = x_byte.float()
            score = dist.log_prob(x_float).sum(dim=2).sum(dim=1)
            if self.sample_probs: return p, score
            else: return x_byte, score 

elif args.px=="pixelcnn":
    from pixelcnn.model import PixelCNN
    from pixelcnn.utils import softmax_loss_1d, sample_from_softmax_1d
    class _Px(nn.Module):
        def __init__(self):
            super().__init__()
            if args.nLatent!=0:
                self.lin1 = nn.Linear(args.nLatent, args.nr_filters*2)
                self.lin2 = nn.Linear(args.nLatent, args.nr_filters*2)
                self.lin3 = nn.Linear(args.nLatent, args.nr_filters*2)
            self.t = Parameter(torch.zeros(1))
            self.pixelcnn = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                                input_channels=1,
                                nr_softmax_bins=2, mode="softmax")

        def forward(self, i, c, x=None):
            if args.nLatent==0:
                cond_blocks=None
            else:
                c = self.t.new_tensor(c)
                cond_blocks = {}
                cond_blocks[(28, 28)] = self.lin1(c)[:, :, None, None]
                cond_blocks[(14, 14)] = self.lin2(c)[:, :, None, None]
                cond_blocks[(7, 7)] = self.lin3(c)[:, :, None, None]

            if x is None:
                x, dist = self.sample(len(i), cond_blocks=cond_blocks)
                _x = x[:,0,:,:]>0
                return _x, -softmax_loss_1d(x, dist)
            else:
                _x = t(x, self.t)[:,None,:,:].float()*2-1
                dist = self.pixelcnn(_x, cond_blocks=cond_blocks, sample=False)
                return x, -softmax_loss_1d(_x, dist)

        def sample(self, sample_batch_size, cond_blocks=None):
            print("sampling pixelcnn...")
            data = self.t.new_zeros(sample_batch_size, 1, 28, 28)
            for i in range(28):
                for j in range(28):
                    with torch.no_grad():
                        data_v = torch.autograd.Variable(data)
                    out   = self.pixelcnn(data_v, sample=True, cond_blocks=cond_blocks)
                    out_sample = sample_from_softmax_1d(out)
                    data[:, :, i, j] = out_sample.data[:, :, i, j]
            return data, out 


elif args.px=="mlp":
    class _Px(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = MLP(args.nLatent)
            self.t = Parameter(torch.ones(1))
            self.sample_probs = False

        def forward(self, i, c, x=None):
            c_tens = self.t.new_tensor(c)

            logits = self.f(c_tens)
            
            dist = Bernoulli(logits = logits)
            if x is None:
                x_float = dist.sample() 
                x_byte = x_float.byte()
            else:
                x_byte = t(x, self.t).byte()
                x_float = x_byte.float()
            
            score = dist.log_prob(x_float).sum(dim=2).sum(dim=1)
            if self.sample_probs: return logits.sigmoid(), score
            else: return x_byte, score 

Px = lambda: PxIID(_Px(), args.nSupport)


def print_weights(model, n_idxs=8, offset=None):
    if offset is None:
        n_idxs = min(n_idxs, nContentLatent)
        #idxs = torch.randperm(nContentLatent).cuda()[:n_idxs] #LT
        idxs = torch.randperm(nContentLatent)[:n_idxs]
    else:
        n_idxs = min(n_idxs, nContentLatent-offset)
        if args.pc == "bernoulli":
            ps = model.prior.logits[:nContentLatent].sigmoid()
            sorted, all_idxs = torch.sort(ps)
            idxs = all_idxs[offset:offset+n_idxs]
            print("ps:", ps[idxs].tolist())
        else:
            idxs = torch.arange(n_idxs)+offset
    #bias = model.decoder.px.b.reshape(1,28,28)

    weights = model.decoder.px.v[idxs].reshape(-1,28,28)
    #weights_norm = (weights - model.decoder.px.v.min())/(model.decoder.px.v.max()-model.decoder.px.v.min())
    weights_norm = weights.clone()
    for i in range(len(weights)):
        weights_norm[i] -= weights_norm[i].min()
        weights_norm[i] /= weights_norm[i].max()
    #bias_norm = (bias - bias.min())/(bias.max()-bias.min())

    #blocks = bias_norm.tolist() + weights_norm.tolist()
    blocks = weights_norm.tolist()
    print(blocks_to_str(blocks))

def blocks_to_str(blocks, chunk=None):
    if chunk is not None:
        return "\n".join(
                blocks_to_str(blocks[i:i+chunk])
                for i in range(0, len(blocks), chunk)
                )
    chars = ["  ", 
             #" .", 
             #"'.", 
             "# ", 
             " ■", 
             #"■#", 
             "■■",
             "█■",
             #"██",
             ]
    def getUnicode(val):
        val = min(1, max(0, float(val)))
        i = int(val*len(chars)) if val<1 else -1

        return chars[i]
        #if val<0.25: return "  "
        #elif val<0.5: return "□□"
        #elif val<0.75: return "▤▤"
        #else: return "■■"

    block_strs = [b[1] if type(b) is tuple else "" for b in blocks]
    blocks = [b[0] if type(b) is tuple else b for b in blocks]
    s = " " + " ".join(("" if blockstr=="" else " " + blockstr + " ").center(28*2, "-") for blockstr in block_strs) + "\n" + \
        "\n".join("|" + "|".join(
                "".join(getUnicode(xx) for xx in block[row_idx])
                for block in blocks) + "|"
            for row_idx in range(28)    
        ) + "\n" + " " + ("--"*28 + " ")*len(blocks) + "\n"
            #for row_idx in range(28))
            #    "".join(getUnicode(dd) for dd in d_row) + 
            #    "|" +
            #    "".join(getUnicode(xx) for xx in x_row) + 
            #    "|" +
            #    "".join(getUnicode(pp) for pp in p_row) + 
            #    "|"
            #    for (d_row, x_row, p_row) in zip(d,x,p)) + "\n" + \
        #" " + "--"*28 + " " + "--"*28 + " " + "--"*28 + " "
    return s

def imFromTensor(xx):
    if len(xx.size())==2:
        xx = 1-xx.clamp(0,1) #improve print contrast
    return Image.fromarray((xx.cpu().detach().numpy() * 255).astype(np.uint8))

def getWeightsImages(model):
    weights = model.decoder.px.v.reshape(-1,28,28)
    weights_norm = (weights - weights.min())/(weights.max()-weights.min())
    return weights_norm

def showWeights(model):
    weights_norm = getWeightsImages(model)
    path = "results/weights"
    shutil.rmtree(path, ignore_errors=True) 
    os.makedirs(path, exist_ok=True)
    for j,xx in enumerate(weights_norm):
        result = imFromTensor(xx)
        f = path + '/%d.png' % j
        result.save(f)


def showDerivations(model, n, save=False):
    assert("turtle" in args.px or "arc" in args.px)
    i = list(range(n))
    cs, _ = model.sample(i, data[:n])

    px = model.decoder.px
    x_steps = {}
    for truncate_to in range(1,px.nStrokes):
        model.decoder.px.truncate_to=truncate_to
        x, _ = model.decoder.px(i, cs, sample_probs=True)
        model.decoder.px.truncate_to=None
        x_steps[truncate_to]=x

    if save:
        print("Saving derivations")
        path = "results/derivation"
        shutil.rmtree(path, ignore_errors=True) 
        os.makedirs(path, exist_ok=True)
        x_prev = {}
        for truncate_to, x_step in x_steps.items():
            for ii,xx in zip(i, x_step):
                if ii not in x_prev or not torch.equal(x_prev[ii], xx):
                    result = imFromTensor(xx)
                    result.save(path + '/%d-%d.png' % (ii,truncate_to))
                x_prev[ii] = xx

        for ii in i:
            result = imFromTensor(data_smooth[ii][0])
            result.save(path + '/%d-target.png' % ii)

def showStrokes(model):
    print("Saving strokes")
    path = "results/strokes"
    shutil.rmtree(path, ignore_errors=True) 
    os.makedirs(path, exist_ok=True)
    strokes = model.decoder.px.get_all_strokes()
    for i in range(len(strokes)):
        result = 1-strokes[i].clamp(0,1) #improve print contrast
        result = torch.cat([result[None]]*3, dim=0)
        result[0, 13:15, 13:15]=1
        result[1:, 13:15, 13:15]=0 #red color
        result = result.transpose(0,2)
        result = imFromTensor(result)
        #return Image.fromarray((xx.cpu().detach().numpy() * 255).astype(np.uint8))
        result.save(path + '/%d.png' % i)

    

def complete(model):
    showDerivations(model, 100, save=True)


#def showDerivations(model, n, save=False, random=False):
#    #weights_norm = model.decoder.px.v.reshape(-1,28*28).sum(dim=1)
#    #sorted, latent_order = torch.sort(weights_norm, descending=True)
#    latent_order = range(nContentLatent)
#
#    for i in range(n):
#        if random: i = np.random.randint(len(data))
#        print("i =", i)
#        #cs, _ = model.encoder([i], data[i:i+1])
#        cs, _ = model.sample([i], data[i:i+1])
#        c = cs[0]
#        c_steps = []
#        c_step_idxs = []
#
#
#        curr_c = [0]*len(c)
#        for j in latent_order:
#            curr_c[j] = c[j]
#            if c[j]==1:
#                c_steps.append([cc for cc in curr_c])
#                c_step_idxs.append(j)
#        c_steps.append(c)
#        c_step_idxs.append(len(c)-1)
#
#        model.decoder.px.sample_probs=True
#        x, _ = model.decoder.px([i]*len(c_steps), c_steps)
#        model.decoder.px.sample_probs=False
#
#        x_list = [(xx.tolist(), "TRANSFORM" if jj==len(c)-1 else str(jj)) for xx,jj in zip(x, c_step_idxs)]
#        print(blocks_to_str(data[i:i+1, 0].tolist() + x_list[-8:], chunk=10))
#    
#        if save:
#            path = "results/derivation/%d" % i
#            shutil.rmtree(path, ignore_errors=True) 
#            os.makedirs(path, exist_ok=True)
#            for idx,(xx,jj) in enumerate(zip(x, c_step_idxs)):
#                result = imFromTensor(xx)
#                #result.save(path + '/%d.png' % jj)
#                result.save(path + '/%d.png' % idx)
#
#            result = imFromTensor(data[i][0])
#            result.save(path + '/target.png')
#
#            yes_idxs = [ii for ii in range(args.nLatent - args.nSpatialLatent) if c[ii]==1]
#            weights_norm = getWeightsImages(model)
#            weights_norm = 1-weights_norm[:,:,:,None].expand(-1,28,28,3)
#            
#            o = torch.ones(weights_norm.size(0), 30,30,3)
#            weights_norm[:, 0, :, :]=0
#            weights_norm[:, -1, :,:]=0
#            weights_norm[:, :, 0, :]=0
#            weights_norm[:, :, -1,:]=0
#            weights_norm[yes_idxs, 0, :, 0]=1
#            weights_norm[yes_idxs, -1, :,0]=1
#            weights_norm[yes_idxs, :, 0, 0]=1
#            weights_norm[yes_idxs, :, -1,0]=1
#            o[:,1:-1,1:-1,:]=weights_norm
#            weights_norm = o
#            weights_norm = torch.cat([xx for xx in weights_norm], dim=1)
#            result = imFromTensor(weights_norm)
#            result.save(path + '/weights.png')
            


def showPrior(model, n=64):
    model.decoder.px.sample_probs=True
    c_prior, x_prior = model.sample_prior([0]*n)
    path = "results/prior"
    shutil.rmtree(path, ignore_errors=True) 
    os.makedirs(path, exist_ok=True)
    for j,xx in enumerate(x_prior):
        result = imFromTensor(xx[0])
        f = path + '/%d.png' % j
        result.save(f)

    if math.sqrt(n)==int(math.sqrt(n)):
        w = int(math.sqrt(n))
        rows = [x_prior[j:j+w] for j in range(0,n,w)]
        im = torch.cat(
                [torch.cat([xx for xx in row], dim=2) for row in rows],
                dim=1)[0]
        result = imFromTensor(im)
        f = path + '/all.png'
        result.save(f)







def iteration(model):
    print_every = 500 if args.px=="pixelcnn" else 100
    if model.iteration==1 or model.iteration%print_every==0:
        i = np.random.randint(len(data))
        c, x = model.sample([i], data[i:i+1], sample_probs=True)
        c_prior, x_prior = model.sample_prior([i])
        d = data[i][0].tolist() 
        x = x[0][0].tolist()
        p = x_prior[0][0].tolist()
        s = blocks_to_str([d,x,p])
        print(s)
        if hasattr(model.decoder.px, "v"): print_weights(model)
        
    #if model.iteration==1 or model.iteration%500==0:
    #    showDerivations(model, n=1, random=True)
    if model.iteration==1 or model.iteration%25000==0:
        evaluate(model)

def getBatch(batch_size, data):
    idxs = np.random.choice(len(data), size=batch_size)
    return idxs, [data[i] for i in idxs]

def getReconstructionError(model):
    errs = []
    for i in range(10):
        i = list(range(100))
        x = data[:100]
        c, _ = model.sample(i, x)
        _, score = model.decoder(i, c, x)
        err =  score.mean().item()
        errs.append(err)
    err = sum(errs)/len(errs)
    print("Reconstruction error:", err)
    with open("reconstructionerror.txt", "w") as f:
        f.write(str(err))

def evaluate(model):
    
    def get_marginal(data):
        b=100
        s = None
        for i in range(0, len(data), b):
            m = model.marginal(range(i,i+b), [data[j] for j in range(i,i+b)], k=300, print_every=100).sum().item()
            s = m if s is None else s+m
        return s / len(data)
        #return s / len(data)



    print("Calculating Train Marginal")
    try:
        train_marginal = get_marginal(data[:100] if model.iteration==1 else data[:1000])
        print("Calculating Test Marginal")
        test_marginal = get_marginal(testdata[:100] if model.iteration==1 else testdata[:1000])
        os.makedirs("results", exist_ok=True)
        with open('results/marginal.txt', 'w') as f:
            string = "Iteration: %d" % model.iteration + " | Train marginal: %3.3f" % train_marginal +  " | Test marginal: %3.3f" % test_marginal
            print(string)
            f.write(string)

        if hasattr(model.decoder.px, "v"):
            showWeights(model)
            showDerivations(model, 20, save=True)
    except Exception as e:
        print("Oh dear...")
        raise e

    showPrior(model)



if __name__=="__main__":
    pc=Pc()
    px=Px()
    rc=Rc()

    x = data[:10]
    i = list(range(10))

    print("testing rc")
    c, score = rc(i, x)
    print("c", c)
    print("score", score)

    print("testing pc")
    c, score = pc(i, c)
    print("c", c)
    print("score", score)

    print("testing px")
    x, score = px(i, c)
    print("x", x)
    print("score", score)

