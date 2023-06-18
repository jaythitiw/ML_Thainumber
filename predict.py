# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 13:42:37 2023

@author: Chalermwong
"""
from pycaret.classification import *
from sklearn.pipeline import Pipeline
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import statistics
import os
from collections import OrderedDict 

os.chdir(os.getcwd())

def fn_predict(imgpath):
    list_model = pd.read_csv('tmp/model_version.csv')

    # list_model = list_model.loc[list_model['ModelVersion'] == list_model['ModelVersion'].max()].sort_values(['Accuracy'],
    #                                                                                                         ascending = False).reset_index(drop=1)

    list_model = list_model.sort_values(['Accuracy'],ascending = False).reset_index(drop=1)                                                                                                   
    list_model = list_model.iloc[:5]

    digit_model = []

    for i in range(list_model.shape[0]):
        digit_model.append( load_model("model/"+str(list_model['ModelVersion'][i])+"_"+list_model['Model'][i]) )

    #load pycaret model
    # digit_model = load_model("model/1_Logistic Regression")
    # imgpath = 'Dataset/0/0a8e39b3-4d7c-4050-a25d-746a6b8de686.png'


    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, gb = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)  # Define the dilation kernel
    dilated = cv2.dilate(gb, kernel, iterations=1)
    # Find contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image
    cropped_image = img[y:y+h, x:x+w]

    resized_image = cv2.resize(cropped_image, (28, 28))

    resized_image
    cv2.imwrite('Dataset_tmp/tmpForPredict.png', resized_image)
    #########
    img = Image.open('Dataset_tmp/tmpForPredict.png').convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28,28),Image.Resampling.LANCZOS)
    pixel = np.array(img)
    pixel = pixel/255.0*16
    pixel = pixel.astype('int')
    my_digit = pixel.reshape(1,-1)[0]
    df = pd.DataFrame([my_digit]).reset_index(drop=True)

    list_result = []
    list_prop = []
    list_score = [[],[],[],[],[]]
    for idx, i in enumerate( digit_model):
    
    # predict_num = predict_model(i, data=df)  #pycaret prediction method
        predict_num = predict_model(i, raw_score = True, data=df)
        # print (predict_num['prediction_score'])
        result = predict_num.at[0,'prediction_label']
        list_result.append(result)
        
        try:
            prop = predict_num.at[0,'prediction_score_'+str(result)]
            list_prop.append(prop)
            for ii in range (10):
                # print (ii)
                list_score[idx].append( float(predict_num.at[0,'prediction_score_'+str(ii)]) )
        except:
            list_prop.append(1.0)
            for ii in range (10):
                if ii == result:
                    list_score[idx].append(1.0)
                else:
                    list_score[idx].append(0.0)

    print (statistics.mode(list_result),list_result)
    result_mode = statistics.mode(list_result)
    
    list_idx = []
    for idx, result in enumerate (list_result):
        if result_mode == result:
            list_idx.append(idx)
    
    list_modelSelect = []
    prop = 0
    score = [0,0,0,0,0,0,0,0,0,0]

    for i in list_idx:
        list_modelSelect.append(list_model['Model'][i])
        # print(i)
        prop = prop+list_prop[i]
        for ii in range (10):
            score[ii] = score[ii]+list_score[i][ii]

    prop = (prop/len(list_idx))
    for ii in range (10):
        score[ii] = score[ii]/len(list_idx)
    
    
    list_modelSelect = list(OrderedDict.fromkeys(list_modelSelect)) 
    
    
    return [result_mode,
            prop,
            list_modelSelect,
            list_model['F1'].mean()*100,
            list_model['Accuracy'].mean()*100,
            score]

# fn_predict('Dataset/0/0a8e39b3-4d7c-4050-a25d-746a6b8de686.png')








      
      
      