import argparse
import itertools
import json
import os
from hashlib import blake2b

import numpy as np
from random import choice
import pickle
import matplotlib.pyplot as plt
from math import *
import scipy
from tqdm import tqdm
import copy

from lag_test_utils_leaky_mod import *
from scipy.optimize import root_scalar

####################ALL PARAMETER COMBINATIONS############################

alphas = list(10**np.linspace(-3, 1, 9)) + ["max"]
# qs = list(np.linspace(0.0, 0.3, 11))
qs = list(np.round(np.linspace(0.0, 0.5, 11), 2))
D = 100.0
pfracs = [0.5]
tau0s = [0.3]
randomness = [True]

# make quick simple run
# alphas = alphas[1:][::2]
# qs = list(np.round(np.linspace(0.0, 0.3, 6), 2))
# D = 100.0
# pfracs = [0.5, 0.7]
# tau0s = [0.3, 0.5]

all_params = []

nRs = [3, 4]
for tau0 in tau0s:
    for a in alphas:
        for pfrac in pfracs:
            for q in qs:
                for r in randomness:
                    for nR in nRs:
                        all_params.append([a, q, pfrac, D, tau0, r, nR])

# # for the effects of D
# nR, randomness, tau0, pfrac = 2, True, 0.3, 0.5
# for a in alphas:
#     for q in qs:
#         for D in [10, 1000, 10000]:
#             all_params.append([a, q, pfrac, D, tau0, randomness, nR])
# print(qs)

# # for the effects of p
# nR, randomness, tau0, D = 2, True, 0.3, 100
# ps = [0.7, 0.3, 0.1]
# for a in alphas:
#     for q in qs:
#         for pfrac in ps:
#             all_params.append([a, q, pfrac, D, tau0, randomness, nR])

# # for the effects of tau0
# nR, randomness, D, pfrac = 2, True, 100, 0.5
# tau0s = [0.1, 0.5, 0.7, 0.9]
# for a in alphas:
#     for q in qs:
#         for tau0 in tau0s:
#             all_params.append([a, q, pfrac, D, tau0, randomness, nR])

ALL_COMBOS = all_params

###########################################################################

REPETITIONS = 1

p_mutate = 1/20
p_mutate_pref = 1/20
phi_lo = 1e-4
phi_hi = 1
log_step = 0.2
T_dilute = 24
gC = 1.0 # deprecated

def phiMutate(phi_now, phi_hi=phi_hi, phi_lo=phi_lo, log_step=log_step):
    # Initial state jumps directly to phi_lo
    if phi_now < phi_lo:
        return phi_lo
    # Convert to logarithmic space for uniform steps
    log_now = np.log10(phi_now)
    # 50% probability for up/down mutation
    direction = np.random.choice([1, -1]) 
    new_log = log_now + direction * log_step
    # Handle boundary conditions
    if new_log > np.log10(phi_hi):
        return phi_hi
    elif new_log < np.log10(phi_lo):
        return MIN_PHI  # Reset to near-zero value
    else:
        return 10**new_log
    
def rRandom(nR, alpha, cycle):
    zero_thr = 1e-8
    if(alpha==0):
        result = choice([i for i in np.eye(nR)])
    elif(alpha=="max"):
        result = np.array([0.5 for i in range(nR)])
    else:
        result = np.round(np.random.dirichlet([alpha]*nR), 10)
        if(np.isnan(result).any() or np.isinf(result).any()):
            result = choice([i for i in np.eye(nR)])
    if(0 in result):
        result = (1-len(result)*zero_thr)*result+zero_thr
    return result

# this part is unchanged so we'll not mention it
###################
from scipy.special import beta, betainc
def minExpectation(alpha): # expectation of min(R1, R2) for dirichlet(alpha, alpha)
    if(alpha=="max"):
        return 0.5
    elif(alpha==0):
        return 0.0
    else:
        integral = betainc(alpha+1, alpha, 0.5)
        return 2 * integral * beta(alpha+1, alpha) / beta(alpha, alpha)
def rProtocol(nR, alpha, cycle): # now only work for nR=2
    zero_thr = 1e-8
    k = 1 
    x1 = minExpectation(alpha)
    Rlist = [np.array([x1, 1-x1]) for i in range(k)] + [np.array([1-x1, x1]) for i in range(k)]
    result = Rlist[ cycle%len(Rlist) ]
    if(0 in result):
        result = (1-len(result)*zero_thr)*result+zero_thr
    return result
#########################

def trial(params, repetition):

    alpha, q, pfrac, D, tau0, randomness, nR = params
    q = 1-(1-q)**(1/(nR-1))
    gs = []
    b_specs = []
    for res in range(nR):
        gi = np.array([0.8*pfrac*(1-q)**i for i in range(nR)])
        gi[res] = gi[res]/pfrac
        gs.append(gi)
        bi = LeakySeqUt(phi=MIN_PHI, g_enz=gi, gC=gC, pref_list=get_rand_pref(res+1, nR), tau0=tau0, biomass=0.01, id=f"{res}_spec{res}")
        # bi = LeakySeqUt(phi=MIN_PHI, g_enz=gi, gC=gC, pref_list=get_smart_pref(gi), tau0=tau0, biomass=0.01, id=f"{res}_spec{res}")
        b_specs.append(bi)

    spc_list = b_specs
    for species in spc_list:
        species.RezeroLag()

    N_trials = 100
    N_cycles = 10000
    all_last_10 = []
    for trial in range(N_trials):
        C = EcoSystem(spc_list)
        b_list, id_list = [], []
        last_10 = []
        mutant_count = nR

        for i in tqdm(range(N_cycles)):
            ######################          here switches random or seasonal          #################################
            if not randomness:
                Rs = rProtocol(nR, alpha, i) 
            else:
                Rs = rRandom(nR, alpha, i)
            C.OneCycle(Rs, T_dilute)
            b_list.append(C.last_cycle['bs'][-1])
            id_list.append(C.last_cycle['ids'])
            # print(C.last_cycle['ids'], C.last_cycle['bs'][-1])
            C.MoveToNext(D)
            if(np.random.rand()<p_mutate):
                # generate invader
                if(len(C.species)>0):
                    spc = np.random.choice(C.species)
                else:
                    spc = np.random.choice(b_specs)
                phi_mut = phiMutate(spc.phi, log_step=np.random.uniform(0.1, 2), phi_hi=1)
                b_mu = LeakySeqUt(phi=phi_mut, g_enz=spc.g, gC=gC, 
                        pref_list=spc.pref, tau0=tau0, biomass=0.01, id=f"{spc.id[0]}_seq{mutant_count}")
                if(np.random.rand()<p_mutate_pref):
                    b_mu = random.choice([i for i in b_specs])
                b_mu.RezeroLag()

                # check if there's an identical species in the community
                if(not any(spc.cat=="Seq" and spc.phi == b_mu.phi and spc.pref == b_mu.pref for spc in C.species)):
                    C.Invade(b_mu)
                    mutant_count += 1

            last_10.append([C.species, C.last_cycle])
            last_10 = last_10[-10:]
        all_last_10.extend(last_10)

    print("[a, q, pfrac, D, tau0, r, nR]", params)
    print([spc.phi for spc in last_10[-1][0] if spc.id in C.last_cycle["ids"]])
    print(last_10[-1][1]["bs"][-1], last_10[-2][1]["bs"][-1])
    print(C.last_cycle)


    if not randomness:
        pickle.dump(all_last_10, open(f"data/with_leaky_1/evolution_with_cout_a={alpha}_q={q}_p={pfrac}_D={D}_tau0={tau0}_seasonal_nR={nR}_2.pkl", "wb"))
    else:
        pickle.dump(all_last_10, open(f"data/with_leaky_1/evolution_with_cout_a={alpha}_q={q}_p={pfrac}_D={D}_tau0={tau0}_random_nR={nR}_2.pkl", "wb"))

def main(task_id):
    total_tasks = len(ALL_COMBOS) * REPETITIONS
    
    # check task_id range
    if task_id >= total_tasks:
        raise ValueError(f"Task ID {task_id} exceeds maximum {total_tasks-1}")
    
    combo_idx = task_id // REPETITIONS
    rep_idx = task_id % REPETITIONS
    params = ALL_COMBOS[combo_idx]
    trial(params, rep_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    args = parser.parse_args()
    main(args.task_id)