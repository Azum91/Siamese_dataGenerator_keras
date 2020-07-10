# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 01:36:37 2020

@author: Azum
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 12:41:01 2020

@author: Azum
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.applications.vgg16 import preprocess_input

class batch_generator_list:
    
    
    def __init__(self,IMAGE_DIR,hp_list,hn_list,width,height,preprocesses):
        
        self.hp_list = hp_list
        
        self.hn_list = hn_list
        
        self.width = width
        
        self.height = height
        
        self.IMAGE_DIR = IMAGE_DIR
        
        self.preprocesses = preprocesses
        
    def concante_list(self):
        
        hp_list = np.array(self.hp_list)
        
        hn_list = np.array(self.hn_list)
        
        concate_list = np.concatenate([hp_list,hn_list],axis=0)
        
        np.random.shuffle(concate_list)
        
        return concate_list
    
    def cached_imread(self,image_path, image_cache):
        if image_path not in image_cache:
            image = plt.imread(image_path)
            image = resize(image, (self.width, self.height))
            image_cache[image_path] = image
        return image_cache[image_path]
    
    def preprocess_images(self,image_names, seed, datagen, image_cache):
        np.random.seed(seed)
        X = np.zeros((len(image_names), self.width, self.height, 3))
        for i, image_name in enumerate(image_names):
            image = self.cached_imread(os.path.join(self.IMAGE_DIR, image_name), image_cache)
            X[i] = datagen.random_transform(image)
        return X
    
    def data_generator(self, batch_size = 16):
        
        concate_list = self.concante_list()
        
        datagen_left = ImageDataGenerator(preprocessing_function=self.preprocesses)
        datagen_right = ImageDataGenerator(preprocessing_function=self.preprocesses)
        image_cache = {}
    
        while True:
             # loop once per epoch
            
            num_recs = len(concate_list)
            
            indices = np.random.permutation(np.arange(num_recs))
            
            num_batches = num_recs // batch_size
            
            for bid in range(num_batches):
                # loop once per batch
                
                batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
                
                batch = [concate_list[i] for i in batch_indices]
                
                seed = np.random.randint(low=0, high=1000, size=1)[0]
                
                Xleft = self.preprocess_images([b[0] for b in batch], seed, 
                                      datagen_left, image_cache)
                Xright = self.preprocess_images([b[1] for b in batch], seed,
                                       datagen_right, image_cache)
                
                input_image = np.concatenate([Xleft,Xright],axis=2)
                
                Y = np_utils.to_categorical(np.array([b[2] for b in batch]))
                
                yield input_image, Y
