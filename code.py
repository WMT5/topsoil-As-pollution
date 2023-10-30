# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 18:06:01 2022
"""
import tensorflow
import keras
import scipy.stats
import pandas as pd
import numpy as np
from keras.layers import Dense,Activation,Input,BatchNormalization,Dropout
from keras.models import Sequential,Model
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score#R square

# data loading

data= np.genfromtxt('Hm-reflect.csv',encoding='utf-8',delimiter=',')  
print(data.shape)

R_Trainresult = []
R_Valresult = []
R_Testresult = []
R2_Trainresult = []
R2_Valresult = []
R2_Testresult = []
MSE_Trainresult = []
MSE_Valresult = []
MSE_Testresult = []
MAE_Trainresult = []
MAE_Valresult = []
MAE_Testresult = []

for j in range(0,12):
    np.random.shuffle(data)
    Feature=data[:,1:4201]
   
    def SNV(Feature):
        m = Feature.shape[0]
        n = Feature.shape[1]
        print(m, n)  #
        Feature_std = np.std(Feature, axis=1)  
        Feature_average = np.mean(Feature, axis=1)  
        Feature_snv = [[((Feature[i][j] - Feature_average[i]) / Feature_std[i]) for j in range(n)] for i in range(m)]
        return  Feature_snv

    Feature1=SNV(Feature)
    Feature1=np.array(Feature1)

    from sklearn import preprocessing
    Fte=preprocessing.scale(Feature1)
    print(Fte.shape)

    #################### Dataset split
    Xtrain=Fte[:14940,:]
    Xval=Fte[14940:16808,:]
    Xtest=Fte[16808:18675,:]

    ytrain=data[:14940,4201] 
    yval=data[14940:16808,4201]
    ytest=data[16808:18675,4201]

    ytrain=np.reciprocal(ytrain)
    yval=np.reciprocal(yval)
    ytest=np.reciprocal(ytest)

    print(Xtrain.shape)
    print(Xval.shape)
    print(Xtest.shape)
    print(ytrain.shape)
    print(yval.shape)
    print(ytest.shape)
    
    Xtrain = Xtrain.astype('float32')
    ytrain = ytrain.astype('float32')
    Xtest = Xtest.astype('float32')
    ytest = ytest.astype('float32')
    Xval = Xval.astype('float32')
    yval = yval.astype('float32')

    inputDims = Xtrain.shape[1]
    print('inputDims:',inputDims)
    EncoderDims = 2048 

 #################### Model construction
    def ae_mlp(inputDims):
        Input_shape = Input(shape=(inputDims,), dtype='float32', name='input')
        AE1 = Dense(EncoderDims, activation='relu')(Input_shape)
        AE1 = Dropout(0.5)(AE1)
        AE2 = Dense(1024, activation='relu')(AE1)
        AE2 = Dropout(0.5)(AE2)
        AE3 = Dense(512, activation='relu')(AE2)
        AE3 = Dropout(0.5)(AE3)
        mlp1 = Dense(200, activation='relu')(AE3) 
        mlp1 = BatchNormalization()(mlp1)
        mlp1 = Activation('relu')(mlp1)
        mlp1 = Dropout(0.45)(mlp1)  
        mlp2 = Dense(150, activation='relu')(mlp1)  
        mlp2 = BatchNormalization()(mlp2)
        mlp2 = Activation('relu')(mlp2)
        mlp2 = Dropout(0.15)(mlp2) 
        output = Dense(1)(mlp2)  
        model = Model(Input_shape, output)
        model.summary()
        return model
    import matplotlib.pyplot as plt
    import os
    result_dir = '/content/drive/MyDrive/Hm/picture/'
    if os.path.isdir(result_dir):
        print('save in :' + result_dir)
    else:
        os.makedirs(result_dir)

    def plot_history(history, result_dir):

        plt.plot(history.history['loss'], marker='.')
        plt.plot(history.history['val_loss'], marker='.')  
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(['loss', 'val_loss'], loc='upper right') 
        plt.savefig(os.path.join(result_dir, 'model_loss_aemlp.png'))
        plt.show

    model = ae_mlp(inputDims)
    model.compile(optimizer='adam',loss='mse')
    history = model.fit(Xtrain,ytrain,validation_data=(Xval,yval),batch_size=512,epochs=700,shuffle=True)
    plot_history(history, result_dir)

  #################### Model evaluation 
    D_pred1 = model.predict(Xtrain)
    
    #R_Trainresult.append(scipy.stats.pearsonr(ytrain,D_pred1)[0])
    R2_Trainresult.append(r2_score(ytrain, D_pred1))
    MSE_Trainresult.append(mean_squared_error(ytrain, D_pred1))
    MAE_Trainresult.append(mean_absolute_error(ytrain, D_pred1))
    
    D_pred2 = model.predict(Xval)
    
    #R_Valresult.append(scipy.stats.pearsonr(ytrain,D_pred2)[0])
    R2_Valresult.append(r2_score(yval, D_pred2))
    MSE_Valresult.append(mean_squared_error(yval, D_pred2))
    MAE_Valresult.append(mean_absolute_error(yval, D_pred2))
   
    D_pred = model.predict(Xtest)
    
    #R_Testresult.append(scipy.stats.pearsonr(ytest, D_pred)[0])
    R2_Testresult.append(r2_score(ytest, D_pred))
    MSE_Testresult.append(mean_squared_error(ytest, D_pred))
    MAE_Testresult.append(mean_absolute_error(ytest, D_pred))

    ytrain=np.squeeze(ytrain)
    D_pred1 =np.squeeze(D_pred1)
    yval=np.squeeze(yval)
    D_pred2 =np.squeeze(D_pred2)
    ytest= np.squeeze(ytest)
    D_pred =np.squeeze(D_pred)
    
    R_Trainresult.append(scipy.stats.pearsonr(ytrain, D_pred1)[0])
    R_Testresult.append(scipy.stats.pearsonr(ytest, D_pred)[0])
    R_Valresult.append(scipy.stats.pearsonr(yval, D_pred2)[0])


np.savetxt('R_Trainresult.csv', R_Trainresult, delimiter = ',')
np.savetxt('R_Valresult.csv', R_Valresult, delimiter = ',')
np.savetxt('R_Testresult.csv', R_Testresult, delimiter = ',')
np.savetxt('R2_Valresult.csv', R2_Valresult, delimiter = ',')
np.savetxt('R2_Trainresult.csv', R2_Trainresult, delimiter = ',')
np.savetxt('R2_Testresult.csv', R2_Testresult, delimiter = ',')
np.savetxt('MSE_Trainresult.csv', MSE_Trainresult, delimiter = ',')
np.savetxt('MSE_Valresult.csv', MSE_Valresult, delimiter = ',')
np.savetxt('MSE_Testresult.csv', MSE_Testresult, delimiter = ',')
np.savetxt('MAE_Trainresult.csv', MAE_Trainresult, delimiter = ',')
np.savetxt('MAE_Valresult.csv', MAE_Valresult, delimiter = ',')
np.savetxt('MAE_Testresult.csv', MAE_Testresult, delimiter = ',')