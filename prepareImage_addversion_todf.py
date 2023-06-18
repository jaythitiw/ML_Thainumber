# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:28:59 2023

@author: Chalermwong
"""

import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

os.chdir(os.getcwd())

data_version = pd.read_excel('tmp/data_version.xlsx')
data_version['rawPath']

data_list = pd.DataFrame()

version = 0 if data_version['version'].shape[0] == 0 else data_version['version'].max() 

for i in range(10):
        raw_path = ('Dataset')
        paths = (f'{raw_path}/{i}')

        # save = ('Dataset_resize')
        resize_path= ('Dataset_resize/')
        dirs = os.listdir(paths)
        # j = 0
        for item in dirs:
            # j +=1
            imgpath = os.path.join(paths, item)
            
            data_list = pd.concat([data_list, pd.DataFrame( {'rawPath':[imgpath],
                                                'resizePath':[resize_path + (f'{i}_{item}')]})], axis =0).reset_index(drop=True) 
            
            
data_list = data_list.loc[~data_list['rawPath'].isin(data_version['rawPath'])].reset_index(drop=1)
data_version = data_version.append(data_list)            
data_version.loc[data_version['version'].isna(), 'version'] = version+1

excel = pd.ExcelWriter('tmp/data_version.xlsx')
data_version.to_excel(excel, index= False)
excel.save()
# excel.close()


data_process = data_version.loc[data_version['version'] == version+1].reset_index(drop=1)

if data_process.shape[0] > 0 :
        
    for i in range (data_process.shape[0]):
        imgpath = data_process['rawPath'][i]
        # print (imgpath)
        
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
        
        cv2.imwrite(data_process['resizePath'][i], resized_image)
            
    # %%
    listdir = data_process['resizePath']
    df = pd.DataFrame()
    for img in listdir:
        imgpaths = img
        images = Image.open(imgpaths).convert('L')
        images = ImageOps.invert(images)
        pixel = np.array(images)
        pixel = pixel/255*16
        pixel = pixel.astype('int')
        digit = pixel.reshape(1,-1)[0]
        digits = [digit]
        df = df.append(digits)
        
    # %%
    y= pd.DataFrame()
    for img in listdir:
        label = img.split('/')[1]
        label = label.split('_')[0] #using 1st deigit of file name as lable (y)
        labels = [label]
        y = y.append(labels)   
    y.rename(columns = {0:'y'}, inplace = True)
    
    # %%
    data_digit = pd.concat([df,y], axis =1).reset_index(drop=True) 
    data_digit = pd.concat([data_digit,data_process['version']], axis =1).reset_index(drop=True) 
    
    # %%
    df_oldversion = pd.read_csv('dataset_version.csv')
    data_digit.columns = df_oldversion.columns
    df_oldversion = df_oldversion.append(data_digit) #pd.concat([df_oldversion,data_digit], axis =0).reset_index(drop=True) 
    df_oldversion.to_csv('dataset_version.csv',index = False)



































