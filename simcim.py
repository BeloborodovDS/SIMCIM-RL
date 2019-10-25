from collections import deque, Counter
import torch
import numpy as np
from gym.spaces import Box, Discrete
from matplotlib import pyplot as plt
import itertools
import networkx as nx

class SIMCIM():
    def __init__(self, A, bias=None, lr=None, momentum=0.0, sigma=0.001, n_iter=1000, device='cpu',
                 batch_size=256, pump_bins=3, lag=10, rshift=0.0, 
                 pump_scale=0.1, reward_kind='rank', continuous=False, span=3, 
                 last_runs=20, percentile=99, add_linear=True,  
                 start_pump=1.0, static_features=True, extra_features=True, curiosity_num=0):
        A = torch.tensor(A, dtype=torch.float32, device=device)
        A.requires_grad = False
        self.A = A
        self.Amean = torch.mean(torch.sum(torch.abs(A), dim=1)).item()
        
        self.batch_size = batch_size
        
        self.pump_bins= pump_bins
        self.lag = lag
        self.rshift = rshift
        self.pump_scale = pump_scale
        self.reward_kind = reward_kind
        self.continuous = continuous
        self.span = span
        self.last_runs = last_runs
        self.percentile = percentile
        self.cut_deque = deque(maxlen=batch_size*last_runs)
        self.add_linear = add_linear
        self.start_pump = start_pump
        self.static_features = static_features
        self.extra_features = extra_features
        self.curiosity_num = curiosity_num
        
        if bias is None:
            self.b = torch.zeros(A.shape[0]).to(A)
        else:
            self.b = torch.tensor(bias).to(A)
        
        self.momentum = momentum
        
        self.n_iter = n_iter
        self.device = device
        self.sigma = sigma
        self.sigma_base = float(np.log10(sigma))

        eival, Q = torch.eig(torch.tensor(A), True)
        eival = eival[:,0]
        eind = torch.argsort(eival, descending=True)
        eival = eival[eind]
        Q = Q[:,eind]
        self.Q = Q
        self.QT = self.Q.transpose(0,1)
        self.eival = eival.to(A)
        self.me = torch.max(eival).item() 
        self.mie = torch.min(eival).item()
        self.asum = torch.sum(A).item()
        self.rec_ind = 0
        self.b1 = torch.mm(self.Q.transpose(0,1), self.b[:,None])[:,0]
        
        index = torch.argmax(eival).item()
        self.eivec = self.Q[:,index].clone()[None,:]
        self.eivec = self.eivec/torch.max(torch.abs(self.eivec))
        
        if lr is None:
            self.get_lr()
            if self.momentum > 0:
                self.lr *= 5
        else:
            self.lr = lr
        self.lr_base = float(np.log10(self.lr))
            
        self.setup_env()
        
        
    def calc_cut(self, c):
        sign = torch.sign(c)
        sign[sign==0] = 1.0
        cut = 0.25*(torch.sum((torch.mm(sign,self.A))*sign, dim=1)+
                    torch.sum(2*sign*self.b[None], dim=1))  - 0.25*self.asum
        return cut
    
    def energy_from_cut(self, cuts, offset):
        eng = -2*(cuts + 0.25*self.asum)+offset
        return eng
    
    def get_lr(self, batch_size=8, n_iter=1000, skew=0.9, spow=1, epow=-2, eps=1e-4):
        base_lr = 10**spow
        decay = (10**(epow-spow))**(1/n_iter)
        sigma = 0.000001
        c = torch.zeros(batch_size, self.A.shape[0], dtype=torch.float32, device=self.device)
        p = - self.me/self.Amean
        stds = []
        lrs = []
        
        for i in range(n_iter):
            lr = base_lr*(decay**i)
            dc = (c*p + torch.mm(c,self.A)/self.Amean + torch.randn_like(c)*(sigma/lr))*lr
            std = torch.mean(torch.abs(dc)).item()
            stds.append(std)
            lrs.append(lr)
            c = torch.clamp(c+dc,-1,1)
            
        lrs = np.array(lrs)
        stds = np.array(stds)
        offset = np.argmax(stds)
        vel = np.diff(np.log(stds)-np.log(lrs))
        minind = np.argmin(vel)
        maxind = minind
        while(vel[maxind]< -eps):
            maxind -= 1
        maxind += 1
        mean = skew*np.log(stds[maxind]) + (1-skew)*np.log(stds[minind])
        ind = np.argmin(np.abs(np.log(stds[offset:])-mean)) + offset
        res = lrs[ind]

        self.lrtest_lr = lrs
        self.lrtest_dc = stds
        self.lrtest_ind = ind
        self.lr = res/self.Amean
        
    def seed(self, sd):
        pass
        
    def setup_env(self):
        self.reward_norm = 1.0
        if self.reward_kind != 'rank':
            self.runpump(self.batch_size)
            print('TEST RUN')
            self.reward_norm = np.abs(np.mean(self.lastcuts))
            if self.reward_norm <=0:
                self.reward_norm = 1.0
        shape = self.A.shape[0]
        self.static_size = 0
        if self.static_features:
            shape += self.A.shape[0]
            self.static_size = self.A.shape[0]
        if self.extra_features:
            shape += 2
        self.observation_space = Box(high=1.0, low=-1, shape=(shape,), dtype=np.float32)
        if self.continuous:
            self.action_space = Box(np.array([-self.span,]), np.array([self.span,]), None, np.float32)
        else:
            self.action_space = Discrete(self.pump_bins)
        self.name = 'SIM CIM'
        self.num_envs = self.batch_size
        self.total_runs = 0
        
        agr = self.Q.abs().sum(dim=0)
        agr = (agr-agr.mean()) / agr.std() / 3
        self.statfeat = agr.repeat(self.num_envs, 1)
        
        self.perc = np.nan

        if self.curiosity_num > 0:
            self.visited = Counter()
        
    def reset(self):
        self.c = torch.zeros(self.batch_size, self.A.shape[0], requires_grad=False, 
                             dtype=torch.float32, device=self.device)
        self.dc = torch.zeros_like(self.c)
        self.cut = self.calc_cut(self.c).cpu().numpy().copy()
        self.p = torch.ones(self.batch_size).to(self.c)*self.start_pump
        self.old_p = torch.ones(self.batch_size).to(self.c)*self.start_pump
        self.i = 0
        self.sumrew = 0
        return self.data2state()
    
    def data2state(self):
        obs = torch.mm(self.c, self.Q)
        if self.static_features:
            obs = torch.cat([self.statfeat, obs], dim=1)
        if self.extra_features:
            t = torch.ones_like(self.p)[:,None] * self.i/self.n_iter
            obs = torch.cat([obs, self.p[:,None], t], dim=1)
        return obs
    
    def action2pump(self, actions):
        actions = actions[:,0]
        if self.continuous:
            delta = torch.clamp(actions, -self.span, self.span)*self.pump_scale/self.span
        else:
            delta = (actions.to(torch.float)*2/(self.pump_bins-1)-1)*self.pump_scale
        p = self.old_p + delta - self.add_linear*self.lag/self.n_iter
        p[p<0]=0
        p[p>1.05]=1.05
        return p
    
    def data2reward(self, extend_perc=True):
        if (self.i>=self.n_iter) and extend_perc:
            self.cut_deque.extend(list(self.cut))
            self.perc = np.percentile(self.cut_deque, self.percentile, interpolation='higher')
            
        if self.reward_kind=='cut':
            r = 100*(self.i>=self.n_iter)*(self.cut/self.reward_norm - self.rshift)
        elif self.reward_kind=='rank':
            if self.i>=self.n_iter:
                gr = (self.cut > self.perc).astype(float)
                le = (self.cut < self.perc).astype(float)
                eq = (self.cut == self.perc).astype(float)
                s = self.percentile/100
                rand = np.random.binomial(1,s,size=eq.shape)
                rand = (1-rand)*s + rand*(1-s)
                r = s*gr - (1-s)*le  
                if np.sum(eq)>0:
                    r -= np.sum(r)*eq/np.sum(eq)

                # curiosity bonus for solutions that are
                # equal to `perc` and occured less than `curiosity_num` times
                if self.curiosity_num > 0:
                    sgn = torch.sign(self.c)
                    sgn[sgn==0] = 1
                    sgn = ((sgn+1)/2).to(dtype=torch.uint8, device='cpu').numpy()
                    curiosity_reward = (-1) * np.ones(self.cut.shape)
                    for j, (ct, res) in enumerate(zip(self.cut, sgn)):
                        if ct == self.perc:
                        #if self.perc-3  <= ct <= self.perc:
                            res = int(''.join(map(str, res)), 2)
                            if self.visited[res] <= self.curiosity_num:
                                curiosity_reward[j] = s
                            self.visited[res] += 1
                    r = np.maximum(r, curiosity_reward)
            else:
                r = np.zeros(self.cut.shape)
        self.sumrew += r
        return torch.tensor(r, device=self.device, dtype=torch.float32).unsqueeze(1)
    
    def step(self, actions, extend_perc=True):
        new_p = self.action2pump(actions)
        self.prev_cut = self.cut
        for j in range(self.lag):
            self.p = (j/self.lag)*new_p + (1-j/self.lag)*self.old_p
            pump = - (self.p*(self.me-self.mie) + self.mie)
            pump = pump[:,None]*self.lr

            newdc = (self.c*pump + 
                     (torch.mm(self.c,self.A) + self.b[None,:])*self.lr + 
                     torch.randn_like(self.c)*self.sigma)
            self.dc = self.dc*self.momentum + newdc*(1-self.momentum)
            ind = (torch.abs(self.c + self.dc) < 1.0).to(self.dc)
            self.c += self.dc*ind
            
            self.i+=1

        self.cut = self.calc_cut(self.c).cpu().numpy().copy()
            
        self.old_p = new_p
        done = (self.i>=self.n_iter)
        flag = done
        
        r = self.data2reward(extend_perc)
        
        info = [dict() for j in range(self.batch_size)]
        done = np.ones(self.batch_size, dtype=bool)*done
        if flag:
            info = [dict(episode=dict(r=_r)) for _r in self.sumrew]

        if flag:
            self.lastcuts = self.calc_cut(self.c).cpu().numpy().copy()
            self.result = torch.sign(self.c).cpu().numpy().copy()
            self.result[self.result==0] = 1
            self.reset()
            self.total_runs += 1
        res = self.data2state(), r, done, info
        return res
        
    def runpump(self, batch_size=128, debug=False, pumpfunc=None):
        self.cut_values = []
        if debug:
            self.trajects = []
            self.nonzero = []
            self.dtrajects = []
            self.maxeiv = []
            self.norm = []
            self.pump = []
            self.gradnorm = []
            
        if pumpfunc is None:
            pumpfunc = lambda it: (1.0 - it/self.n_iter)
            
        c = torch.zeros(batch_size, self.A.shape[0], requires_grad=False, 
                        dtype=torch.float32, device=self.device)
        
        dc = torch.zeros_like(c)
        
        for i in range(self.n_iter):
            if isinstance(pumpfunc, list):
                p = torch.tensor([f(i) for f in pumpfunc]).to(c)[:,None]
            else:
                p = pumpfunc(i)
            p = - (p*(self.me-self.mie) + self.mie)*self.lr
            
            newdc = (c*p + (torch.mm(c,self.A) + self.b[None,:]) * self.lr
                     + torch.randn_like(c)*self.sigma)
            dc = dc*self.momentum + newdc*(1-self.momentum)
                
            ind = (torch.abs(c + dc) < 1.0).to(dc)
            c += dc*ind
            
            cut = self.calc_cut(c)
            self.cut_values.append(cut.cpu().numpy().copy())
            
            if debug:
                rec_ind = self.rec_ind
                
                self.gradnorm.append(torch.sum(torch.abs((newdc*ind)[rec_ind,:])))
                
                self.pump.append(p)

                self.dtrajects.append(torch.mv(self.Q,c[rec_ind,:]).cpu().numpy().copy())
                self.trajects.append(c[rec_ind,:].cpu().numpy().copy())
                nz = torch.sum(ind[rec_ind,:]).item()
                self.nonzero.append(nz)
                Ac = torch.mv(self.A,c[rec_ind,:])
                eiv = torch.sqrt(torch.sum(Ac*Ac))/torch.sqrt(torch.sum(c[rec_ind,:]*c[rec_ind,:]))
                self.maxeiv.append(eiv.item())
                Ac = Ac/torch.sqrt(torch.sum(Ac*Ac))
                Ac = Ac - c[rec_ind,:]/torch.sqrt(torch.sum(c[rec_ind,:]*c[rec_ind,:]))
                Ac = torch.sqrt(torch.sum(Ac*Ac))
                self.norm.append(Ac.item())
                
        self.lastcuts = self.calc_cut(c).cpu().numpy().copy()
                
        self.result = torch.sign(c)
        self.result[self.result==0] = 1
        return self.result
        
    def report(self, draw_traj=True, lim=0):
        if draw_traj:
            maxcuts = np.array(self.cut_values[-1])
            plt.hist(maxcuts, bins=100)
            plt.show()
            print('super max: ', np.max(maxcuts))

            traj = np.array(self.trajects).T
            for i in range(traj.shape[0]):
                plt.plot(traj[i][-lim:])
            plt.show()

            dtraj = np.array(self.dtrajects).T
            for i in range(dtraj.shape[0]):
                plt.plot(dtraj[i][-lim:])
            plt.show()

        plt.plot(self.nonzero[-lim:])
        plt.title('Nonzero')
        plt.show()
        
        try:
            plt.plot(self.pump[-lim:])
            plt.title('pump')
            plt.show()
        except:
            pass

        plt.plot(np.array(self.cut_values)[:,-1][-lim:])
        plt.title('CUT')
        plt.show()

        plt.plot(np.log(np.array(self.maxeiv)[-lim:]/self.me))
        for ev in self.eival.cpu().numpy()[:10]:
            if ev > 0:
                plt.plot([0,1000], [np.log(ev/self.me),np.log(ev/self.me)], '-')
        plt.title('EIV')
        plt.show()

        plt.plot(self.norm[-lim:])
        plt.title('Norm')
        plt.show()


class SIMCollection():
    def __init__(self, envs, baselines=None):
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.device = envs[0].device
        self.name = 'SIM CIM vector'
        self.num_envs = sum([e.num_envs for e in envs])
        self.envs = envs
        if baselines is None:
            self.baselines = np.zeros(len(self.envs))
        else:
            self.baselines = np.array(baselines)
        
    @property
    def perc(self):
        return np.array([e.perc for e in self.envs]) - self.baselines
        
    def reset(self):
        self.index = torch.randperm(self.num_envs, device=self.device)
        self.index_numpy = self.index.cpu().numpy()
        self.inverse_index = torch.argsort(self.index)
        
        states = [e.reset() for e in self.envs]
        states = torch.cat(states)
        
        return states[self.index, :]
    
    def step(self, actions, **kwargs):
        actions = torch.chunk(actions[self.inverse_index, :], len(self.envs))
        
        data = [e.step(a, **kwargs) for e,a in zip(self.envs, actions)]
        states, rewards, dones, infos = list(zip(*data))
        
        states = torch.cat(states)[self.index,:]
        rewards = torch.cat(rewards)[self.index,:]
        dones = np.concatenate(dones)[self.index_numpy]
        infos = list(itertools.chain.from_iterable(infos))
        infos = [infos[i] for i in self.index_numpy]
        
        if np.any(dones):
            states = self.reset()
        
        return states, rewards, dones, infos


class SIMCollectionRandom():
    def __init__(self, size, prob, num_graphs, batch_size, config, scale=-1, device='cuda'):
        self.envs = []
        for _ in range(num_graphs):
            graph = scale * nx.to_numpy_array(nx.erdos_renyi_graph(size, prob))
            sim = SIMCIM(graph, device=device, batch_size=batch_size, **config)
            self.envs.append(sim)
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.device = device
        self.name = 'Random SIM CIM vector'
        self.num_graphs = num_graphs
        self.batch_size = batch_size
        
    @property
    def perc(self):
        perc = np.array([e.perc for e in self.envs])
        perc = np.mean(perc[np.isfinite(perc)])
        return perc
        
    def reset(self):
        self.index = np.random.randint(self.num_graphs)
        return self.envs[self.index].reset()
    
    def step(self, actions):        
        states, rewards, dones, infos = self.envs[self.index].step(actions)
        if np.any(dones):
            states = self.reset()
        return states, rewards, dones, infos
    

class SIMGeneratorRandom():
    def __init__(self, size, prob, batch_size, config, 
                 scale=-1, device='cuda', track_percentiles=100, keep=10, n_sims=4):
        self.size = size
        self.prob = prob
        self.batch_size = batch_size
        self.config = config
        self.scale = scale
        self.device = device
        self.keep = keep
        self.n_sims = n_sims
        
        self.make_sim()
        
        self.observation_space = self.sim.observation_space
        self.action_space = self.sim.action_space
        self.name = 'Random SIM CIM vector'
        
        self.perc_deque = deque(maxlen=track_percentiles)
        
        self.i = 0
        
    def make_sim(self):
        envs = []
        for i in range(self.n_sims):
            graph = self.scale * nx.to_numpy_array(nx.erdos_renyi_graph(self.size, self.prob))
            sim = SIMCIM(graph, device=self.device, 
                         batch_size=self.batch_size//self.n_sims, **self.config)
            envs.append(sim)
        self.sim = SIMCollection(envs)
        
    @property
    def perc(self):
        perc = np.mean(list(self.perc_deque))
        return perc
        
    def reset(self):
        self.i += 1
        if self.i >= self.keep:
            self.i = 0
            self.make_sim()
        return self.sim.reset()
    
    def step(self, actions, **kwargs):        
        states, rewards, dones, infos = self.sim.step(actions, **kwargs)
        if np.any(dones):
            self.perc_deque.extend(list(self.sim.perc))
            states = self.reset()
        return states, rewards, dones, infos