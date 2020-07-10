# Siamese_dataGenerator_keras
datagenerator for siamese neural network through local directory using keras

import pandas as pd
from keras.applications.vgg16 import preprocess_input


#getting positive List format [image_1_name*, image_2_name*, categorical*]
pos_path = '../p_list_path.csv'

df_pos = pd.read_csv(pos_path)

df_pos  = df_pos.loc[:,~df_pos.columns.str.contains('^Unnamed')]

#getting negative List format [image_1_name*, image_2_name*, categorical*]
neg_list_path = '../n_list_path.csv'

df_neg = pd.read_csv(neg_list_path)

df_neg  = df_neg.loc[:,~df_neg.columns.str.contains('^Unnamed')]

# base_path for images all images
images_base_path = '../training'

#Generating list on object call
obj_batchgen = Siamese_batch_generator(base_path,df_pos,df_neg,224,224,preprocess_input) 

#Generate Batch from Object of Siamese_batch_generator

batch_gen = obj_batchgen.data_generator(batch_size=32)

batch_check = next(batch_gen)

print(batch_check.shape)
