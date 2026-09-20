#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 15:14:02 2026

@author: asandhir
"""
import numpy as np
from tqdm import tqdm

class Tracking:
    def __init__(self, itterations):
        self.progress_bar = tqdm(total=itterations, desc="Optimizing", unit="gen", leave = True)
        self.parameter_history = []
        self.parameter_history.insert(0, [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN])
        self.target_prob_history = []
        
    def __call__(self, mutation, convergence):
        # Record current best objective value and parameters
        self.parameter_history.append(np.copy(mutation).astype(int))
        self.progress_bar.update(1)

    def close(self):
        self.progress_bar.close()