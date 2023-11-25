#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 22:17:30 2023

@author: jingyuanzhang
"""

######Monte Carlo Simulation to get stock price


import numpy as np
import pandas as pd
import yfinance as yf

df = yf.download("IBM",
                 start="2021-01-01",
                 end="2022-01-31")

returns = df["Adj Close"].pct_change().dropna()

returns.plot(title = "IBM's returns")

train = returns["2021"]
test = returns["2022"]

T = len(test)
N= len(test)
S_0 = df.loc[train.index[-1], "Adj Close"]
N_SIM = 2
mu = train.mean()
sigma = train.std()
x = np.random.seed(42)

def simulate_gbm(s_0,mu,sigma,n_sims,T,N,randomseed =42):
    np.random.seed(randomseed)
    
    dt = T/N
    dW = np.random.normal(scale = np.sqrt(dt),size = (n_sims,N))
    W = np.cumsum(dW, axis = 1)
    
    time_step = np.linspace(dt,T,N)
    time_steps = np.broadcast_to(time_step,(n_sims,N))
    S_t = (s_0 * np.exp((mu - 0.5 * sigma ** 2)*time_steps +sigma *W))
    S_t = np.insert(S_t, 0, s_0, axis=1)
    
    return  S_t

gbm_simulations = simulate_gbm(S_0, mu, sigma, N_SIM, T,N)
sim_df = pd.DataFrame(np.transpose(gbm_simulations),
                     index = train.index[-1:].union(test.index))

res_df = sim_df.mean(axis = 1).to_frame()
res_df = res_df.join(df["Adj Close"])
res_df.columns = ["simulation_average","adj_close_price"]

ax = sim_df.plot(
    alpha=0.3, legend=False, title="Simulation's results"
)
res_df.plot(ax=ax, color = ["red", "blue"])