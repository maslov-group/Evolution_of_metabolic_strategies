import itertools
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from math import *
import scipy
from tqdm import tqdm
import copy

from utils import *
from scipy.optimize import root_scalar

DIEOUT_BOUND = 1e-8
INVADE_BOLUS = 1e-8
MIN_PHI=1e-200
    
class LeakySeqUt: # sequential utilizer, this time with lags
    def __init__(self, phi, g_enz, gC, pref_list, tau0, biomass, id):
        '''
        growth_Rate_list_single: float array[N_res], generated from gaussian
        pref_list: int array[N_res]
        biomass: float
        '''
        self.cat = "Seq"
        self.id = id
        self.gC = gC # will not touch gC 
        self.phi = phi
        self.alive = True
        self.nr = len(g_enz)
        self.g = g_enz # in principle, gaussian
        self.pref = pref_list
        self.tau0 = tau0
        self.b = biomass
        self.eating = np.array([False for i in range(self.nr)]) # default - not eating anything
        self.lag_from = -1
        self.lag_to = -1
        self.lag_left = 0.0
    def Dilute(self, D):
        '''
        D: float
        '''
        self.b /= D
        if(self.b<DIEOUT_BOUND):
            self.alive = False
    def ReturnEating(self, Rs, consider_lag=True):
        '''
        Rs: float array of all resources
        '''
        nR = len(Rs)
        phi = self.phi
        # phi = phi / nR * (nR-1) #spring
        npres = np.sum(Rs>0)
        eating = np.zeros(self.nr)
        if(npres>1):
            # eating[Rs>0] = phi/(npres-1) #2025spring
            eating[Rs>0] = phi/npres #summer
            # eating[Rs>0] = phi
        if(self.lag_left==0 or consider_lag==False):
            for r in self.pref:
                if(Rs[r-1]>0):
                    if(npres>1):
                        # eating[r-1] = 1 - phi #spring
                        eating[r-1] = 1 - phi*(npres-1)/npres #summer
                        # eating[r-1] = 1 - phi*(npres-1)
                    else:
                        eating[r-1] = 1
                    break
        return eating
    def GetEating(self, Rs):
        '''
        Rs: float array of all resources
        kindof deprecated
        '''
        # self.eating = np.array([False for i in range(self.nr)])
        # if(self.lag_left==0):
        #     for r in self.pref:
        #         if(Rs[r-1]>0):
        #             self.eating[r-1] = True
        #             break
        self.eating = self.ReturnEating(Rs)
    def GetGrowthRate(self):
        if(self.lag_left>0):
            return 0
        else:
            return self.g @ self.eating
    def GetDep(self):
        '''
        In all cases assume yield Y=1
        '''
        if(self.GetGrowthRate()==0):
            return self.eating.astype(float)
        else:
            return (self.g*self.eating) / self.GetGrowthRate()
    def GetTau(self, R_dep, Rs):
        '''
        R_dep: int, the last depleted resource
        Rs: float array of all resources. At this point, R_dep should be 0 in Rs. 
        '''
        nR = len(Rs)
        Rs_before = copy.deepcopy(Rs)
        Rs_before[R_dep-1] = 1.0
        eat_before = self.ReturnEating(Rs_before, consider_lag=False)
        eat_after = self.ReturnEating(Rs, consider_lag=False)
        if(np.sum(eat_after)==0):
            return 0
        # enz_before = max(1-eat_before[R_dep-1], MIN_PHI)
        enz_frac = np.max((eat_after+MIN_PHI) / (eat_before+MIN_PHI))
        return self.tau0*log(enz_frac)
    def GetLag(self, R_dep, Rs): # update lag state after 1 resource gets depleted
        '''
        R_dep: int, the last depleted resource
        Rs: float array of all resources. At this point, R_dep should be 0 in Rs. 
        '''
        if(np.sum(Rs)==0): # if no more resources present, straight up rezero everything
            self.lag_from = -1
            self.lag_to = -1
            self.lag_left = 0.0
            return 0
        if(self.lag_left==0):
            if(self.eating[R_dep-1]>0):
                self.lag_from = R_dep
                for r in self.pref:
                    if(Rs[r-1]>0):
                        self.lag_to = r
                self.lag_left += self.GetTau(R_dep, Rs)
        else:
            if(R_dep==self.lag_to):
                for r in self.pref:
                    if(Rs[r-1]>0):
                        self.lag_to = r
                self.lag_left += self.GetTau(R_dep, Rs)
    ## apply the initial lag in the simplest way
    def RezeroLag(self):
        self.lag_from = self.pref[-1]
        self.lag_to = self.pref[0]
        self.lag_left = 0 # no init lag

class EcoSystem: 
    def __init__(self, species=[]):
        '''
        Rs_init: float array [N_res]
        species: list of species; species are SeqUt, CoUt etc examples
        '''
        self.res = np.array([])
        self.species = species
        for species in self.species:
            species.alive = True
        self.last_cycle = {'ids':[species.id for species in self.species], 'ts':[], 'cs':[], 'bs':[]}
    def OneCycle(self, R0, T_dilute):
        '''
        R0: float array [N_res] added in this cycle
        T_dilute: float, cutoff of dilution time
        At the beginning of each cycle, res are at the scale of 1 and species are at the scale of 1/D
        '''
        self.species = [species for species in self.species if species.alive]
        for spc in self.species:
            if(spc.phi<=2*MIN_PHI):
                spc.RezeroLag()
        ts, cs, bs = [], [], []
        t_switch = 0
        self.res = copy.deepcopy(R0)
        nr = len(R0)
        state_flag = -1 # -1: nothing happens; -2: finished a lag; >=0: a resource is depleted. 
        ts.append(t_switch)
        cs.append(copy.deepcopy(self.res))
        bs.append(np.array([species.b for species in self.species]))
        while t_switch < T_dilute:
            # print(t_switch)
            t_step = T_dilute - t_switch
            for species in self.species:
                species.GetEating(self.res)
                if (species.lag_left>0):
                    t_i = species.lag_left
                    t_step = min(t_step, t_i)
                    state_flag = -2
            for r_id, r in enumerate(self.res):
                if r>0:
                    def remain(t):
                        return r - sum([species.b * (exp(species.GetGrowthRate()*t)-1) * species.GetDep()[r_id] for species in self.species])
                    if remain(t_step)<0:
                        t_i = root_scalar(remain, bracket = [0, t_step], method='brenth').root
                        t_step = t_i
                        state_flag = r_id
            t_switch = t_switch + t_step
            # update the system according to the t_step
            
            if (state_flag==-2): # if it's a lag
                # first update res before species, because we need b at the previous timepoint for res change
                for r_id, r in enumerate(self.res):
                    self.res[r_id] = r - sum([species.b * (exp(species.GetGrowthRate()*t_step)-1) * species.GetDep()[r_id] for species in self.species])
                # update the species abundance and their lag_left
                for species in self.species:
                    if(species.lag_left>0):
                        # if(species.cat=="Cout"):
                        #     species.b = species.b * exp(species.GetGrowthRate()*t_step)
                        species.lag_left = species.lag_left - t_step
                    else:
                        species.b = species.b * exp(species.GetGrowthRate()*t_step)

            elif(state_flag>-1): # if it's a depletion of resource
                for r_id, r in enumerate(self.res):
                    self.res[r_id] = r - sum([species.b * (exp(species.GetGrowthRate()*t_step)-1) * species.GetDep()[r_id] for species in self.species])
                    self.res[state_flag] = 0
                # update species abundance; update lag_left
                for species in self.species:
                    if(species.lag_left>0):
                        species.lag_left = species.lag_left - t_step
                        # if(species.cat=="Cout"):
                        #     species.b = species.b * exp(species.GetGrowthRate()*t_step)
                    else:
                        species.b = species.b * exp(species.GetGrowthRate()*t_step)
                    species.GetLag(state_flag+1, self.res)
                    # print("B:", species.b)
                # print("R:", self.res)
            else:
                for r_id, r in enumerate(self.res):
                    self.res[r_id] = r - sum([species.b * (exp(species.GetGrowthRate()*t_step)-1) * species.GetDep()[r_id] for species in self.species])
                # update species abundance and lag_left
                for species in self.species:
                    species.b = species.b * exp(species.GetGrowthRate()*t_step)
                    if(species.lag_left>0):
                        species.lag_left = species.lag_left - t_step
            state_flag = -1
            ts.append(t_switch)
            cs.append(copy.deepcopy(self.res))
            bs.append(np.array([species.b for species in self.species]))
        self.last_cycle = {'ids':[species.id for species in self.species], 'ts':ts, 'cs': cs, 'bs': bs}
    def MoveToNext(self, D):
        '''
        D: float, dilution rate
        This does not include adding new resources
        '''
        self.res /= D
        for species in self.species:
            species.Dilute(D)
            species.RezeroLag()
    def CheckInvade(self, invader, D):
        '''
        invader: a species
        D: float, dilution rate
        '''
        # TBD
        if(len(self.species) == 0):
            return True
        if(invader.id in [species.id for species in self.species]):
            return False
        ts, cs= self.last_cycle["ts"], self.last_cycle["cs"],
        growth = 0
        invader.RezeroLag()
        for idx, t_pt in enumerate(ts[:-1]):
            if(np.sum(cs[idx])>0):
                delta_t = ts[idx+1] - ts[idx]
                if(invader.lag_left==0):
                    invader.GetEating(cs[idx])
                    growth += invader.GetGrowthRate()*delta_t
                    # update the lag state
                    for r in range(len(self.res)):
                        if(cs[idx][r]!=0 and cs[idx+1][r]==0):
                            invader.GetLag(r+1, cs[idx+1])
                else:
                    if(invader.lag_left >= delta_t):
                        invader.lag_left -= delta_t
                    else:
                        invader.lag_left = 0
                        delta_t -= invader.lag_left
                        invader.GetEating(cs[idx])
                        growth += invader.GetGrowthRate()*delta_t
                        # update the lag state
                        for r in range(len(self.res)):
                            if(cs[idx][r]!=0 and cs[idx+1][r]==0):
                                invader.GetLag(r+1, cs[idx+1])
        return growth>log(D)
    def Invade(self, invader):
        '''
        invader: a species
        '''
        invader.alive = True
        invader.b = INVADE_BOLUS
        self.species.append(invader)

# make some function about plotting the biomass across cycles
def vis_biomass(id_list, blist):
    '''
    id_list: list of list of int; each element is the ID of species that are present at the end of a cycle
    blist: list of np.array of float; each element is the abundance of species at the end of a cycle
    '''
    all_keys = set(sum(id_list, []))
    all_info_dict = {key:[] for key in all_keys}
    for cycle, ids in enumerate(id_list):
        for key in all_keys:
            all_info_dict[key].append(0)
        for idx, id in enumerate(ids):
            all_info_dict[id][-1] = blist[cycle][idx]
    for key in all_info_dict:
        plt.plot(range(len(blist)), all_info_dict[key], label=key)
    plt.xlabel("Dilution cycles")
    plt.ylabel("Species abundance")

def get_rand_pref(i, n): 
    '''
    get a random pref order where i is the top choice among n resources
    '''
    if i < 1 or i > n:
        raise ValueError(f"i must be between 1 and n (inclusive), got i={i}, n={n}")
    elements = list(range(1, n+1))
    elements.remove(i)
    permuted = np.random.permutation(elements)
    return (i,) + tuple(permuted)

def get_smart_pref(g):
    '''
    get the pref order for a smart spc with given g
    '''
    return tuple(np.argsort(g)[::-1]+1)