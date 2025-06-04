"""
Created on Wed Mar 20 21:48:04 2019
Initialization
@author: Stamatis
"""

""" import libraries """
import pandas as pd
import numpy as np
import math
import random
import json
import pickle
import time
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from time import sleep

""" initialize parameters """
if 'run_all' not in globals():
    need_to_update = True
elif run_all == False:
    need_to_update = True
else:
    need_to_update = False
decision_periods, years_in_period = 4, 5
li_price, la_price, vr_price, fw_price = [420, 310, 167, 150], [142, 115, 77, 65], [385, 255, 120, 95], [3100, 2600, 1950, 1700]
tech_price = [li_price, la_price, vr_price, fw_price]
li_life, la_life, vr_life, fw_life = [12, 17, 19, 20], [9, 11, 13, 14], [13, 17, 20, 21], [20, 26, 30, 32]
tech_life = [li_life, la_life, vr_life, fw_life]
li_eff, la_eff, vr_eff, fw_eff = [0.95, 0.96, 0.97, 0.98], [0.80, 0.81, 0.83, 0.84], [0.70, 0.73, 0.78, 0.79], [0.84, 0.85, 0.87, 0.88]
tech_eff = [li_eff, la_eff, vr_eff, fw_eff]
li_dod, la_dod, vr_dod, fw_dod = [0.90, 0.90, 0.90, 0.90], [0.55, 0.55, 0.55, 0.55], [1.00, 1.00, 1.00, 1.00], [0.86, 0.86, 0.86, 0.86]
tech_dod = [li_dod, la_dod, vr_dod, fw_dod]
if need_to_update: b_choices, superposed_scenario = [300, 1000, 3000], False
b_levels, initial_soc, tech = len(b_choices), 1, len(tech_eff)
transition_prob = [[0.7, 0.7, 0.7, 0], [0.7, 0.7, 0.7, 0], [0.7, 0.7, 0.7, 0], [0.7, 0.7, 0.7, 0]]
initial_seed = 12
