#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns 
from PIL import Image, ImageOps

from pycaret.classification import *
from pycaret.datasets import get_data

import time
import os

os.chdir(os.getcwd())

def fn_train_model(cev_percent=95,train_size = 0.8,fix_imbalance = True ):

    digits = pd.read_csv('dataset_version.csv')

    dataversion = 1
    #filter version
    ##digits = digits.loc[digits['version']<= dataversion]

    digits =  digits.iloc[:,:-1].reset_index(drop=1)

    X = digits.iloc[:,:-1]
    y = digits.iloc[:,-1:]

    # cev_percent = 95
    # train_size = 0.8
    
    #Presision for PCA (%)
    # cev_percent = 95  #input as percent then convert to decimal
    cev = cev_percent/100
    pca_num = PCA(cev) 
    pca_num.fit(X)
    n_comp = pca_num.n_components_


    #setup train - test data and preprocessing: SMOTE and PCA 
    clf = setup(data = digits,
                target = y,
                session_id=123,
                imputation_type=None,
                train_size = train_size,
                fix_imbalance = fix_imbalance,
                pca = True,
                pca_method = 'linear',
                pca_components = n_comp,
            ) # use_gpu=True


    ############################
    # allModel = compare_models()
    all_model = compare_models(n_select = 13)
    results = pull()
    results.Model.tolist()
    # results[:5]
    ############################

    df_model_version = pd.read_csv('tmp/model_version.csv')

    model_version = 0 if df_model_version['ModelVersion'].shape[0] == 0 else df_model_version['ModelVersion'].max() 

    results['DataVersion'] = dataversion
    results['ModelVersion'] = model_version+1
    results['cev_percent'] = cev_percent
    results['train_size'] = train_size
    print('check')
    results.to_csv('tmp/model_version.csv', mode='a', index=False, header=False)

    for idx, model in enumerate(results.Model.tolist()):
        save_model(all_model[idx], 'model/'+str(model_version+1)+'_'+model)
    return results['cev_percent'],results['train_size']





# fn_train_model(90,0.7,True)