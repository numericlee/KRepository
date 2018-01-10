# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 22:52:29 2017
@author: Walzer
"""
#Inference Code


from __future__ import print_function, division
import numpy as np  #, os
import tensorflow as tf #needed?
#from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.optimizers import Momentum
import pandas as pd


'''
Generating an Input Pipeline
reads in a minibatch, extracts images and labels,
returns the data in a form that can be easily streamed as a feed dictionary to a TFLearn based CNN.
'''
# input_pipeline(filename, path): prepares a batch of images and labels for training
# parameters:      filename - the file to be batched into the model
#                 path - the folder where filename resides
# returns:         image_batch - a batch of images o train or test on
#                  label_batch - a batch of labels related to the image_batch



def input_pipeline(filename,param):
    minibatched_scans,  image_batch, label_batch,  pzid_batch = [],[],[],[]
    minibatched_scans = np.load(filename) #Load a batch of pre-procd tz scans #revised
    #shuffle(minibatched_scans)     #Shuffle to randomize for input into the model revised
    
    # separate images and labels
    for example in minibatched_scans:
            pzid_batch.append(example[0])
            if param==1:
                label_batch.append(example[1]) #example[1] is a 2 item list
                image_batch.append(example[2])
            else: #param=2
                image_batch.append(example[1]) #example[1] is a 2 item list                
    
    #pad or crop the first dimension
    iblen=[len(x) for x in image_batch]
    N1 = [int(.5* (Dm1 - x)) for x in iblen] #list of dimensions
    #max_len =  max(iblen)
    ib2=[]
    for jqz,scan in enumerate(image_batch):
        if iblen[jqz]<Dm1: #pad
            N2 = Dm1 - iblen[jqz] - N1[jqz]
            scan2= np.pad(scan, pad_width= ((N1[jqz], N2)) ,mode= 'edge')
        elif iblen[jqz]>Dm1: #crop
            N2 = -N1[jqz]+Dm1
            scan2= scan[-N1[jqz]:N2,:]
        else:
            scan2=scan
        ib2.append(scan2)

    #pad or crop the second dimension
    ib2len=[x.shape[1] for x in ib2]
    M1 = [int(.5* (Dm1 - x)) for x in ib2len]
    #max2_len = max(ib2len)
    
    ib3=np.zeros((len(iblen),Dm1,Dm1),dtype=np.float32)
    
    for jqz,scan2 in enumerate(ib2):
        if ib2len[jqz]<Dm1: #pad
            M2 = Dm1 - ib2len[jqz] - M1[jqz]
            scan3 = np.pad(scan2, pad_width= ((0, 0),(M1[jqz], M2)),mode= 'edge')
        elif ib2len[jqz]>Dm1: #crop
            M2 = -M1[jqz]+Dm1
            scan3= scan2[:, -M1[jqz]:M2]
        else:
            scan3=scan2
    
        ib3[jqz,:,:] =scan3

    pzid_batch = np.array(pzid_batch)
    if param==1:
        label_batch = np.asarray(label_batch, dtype=np.float32)
    else: #param=2
        label_batch=pzid_batch*0

    return pzid_batch, label_batch, ib3

def DNN(Dm1):
        network = input_data(shape=[None, Dm1, Dm1, 1], name='imAges')
        
        network = conv_2d(network, 96, 11, strides=4, activation='relu') #layer1 has output of __x96
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        
        network = conv_2d(network, 256, 5, activation='relu') #layer2
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        
        network = conv_2d(network, 384, 3, activation='relu') #layer3
        network = conv_2d(network, 384, 3, activation='relu') 
        network = conv_2d(network, 256, 3, activation='relu') 
        network = max_pool_2d(network, 3, strides=2)
        network = local_response_normalization(network)
        
        network = fully_connected(network, 4096, activation='tanh') #layer4
        network = dropout(network, 0.5)
    
        
        network = fully_connected(network, 4096, activation='tanh') #layer5
        network = dropout(network, 0.5)
    
        #network=tflearn.reshape(network,[-1]) #new
    
        network = fully_connected(network, 2, activation='softmax') #restored
    
        momentum = Momentum(learning_rate=Learning_Rate, lr_decay=0.99, decay_step=1000,staircase=True)    
        #Momentum (momentum=0.9,  use_locking=False, name='Momentum')
        #momentum='momentum'
    
        network = regression(network, optimizer=momentum, loss='categorical_crossentropy',  #layer7
                             learning_rate=Learning_Rate, name='labels')
    
    
        new_model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + ModelName,tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)
        return new_model



def BuildRegrTbl(XP):
    #builds regression table using the 3 highest scroing ML views and the 48 metrics
    global allPZID,prob48,p48,pred,eachID,XP1 #diag
    XP1=XP
    allPZID  = [int(x+.5) for x in  sorted(set( XP[:,0])) ]
    Max3Table = np.zeros( (len(allPZID),10)   ) #new in 82. add column for 48Metric predictions
    Max3Table[:,0] = allPZID
   

    p48N = pd.read_csv(PATH1 + "/predict48NB.csv",delimiter="|").set_index('pzvID')
    prob48 = p48N["prediction"]

    for j,eachID in enumerate(allPZID):
       itemindex = np.where(XP[:,0]==eachID)[0]  #turns list of row indices
       pred = XP[itemindex,   1 ] 
       Max3Table[j,2:5] = pred[np.argsort(-pred)][0:3] 

       try:
           Max3Table[j,8]  = prob48[eachID]
       except:
           print("<BuildRegTbl> error retrieving and storing 48metric pred",j,eachID)
           
    #cross terms
    Max3Table[:,5] =  Max3Table[:,2] * Max3Table[:,3]  
    Max3Table[:,6] =  Max3Table[:,2] * Max3Table[:,4]  
    Max3Table[:,7] =  Max3Table[:,3] * Max3Table[:,4]  
    
    return Max3Table



def viewCombiner2(Threat_Rgn)  :
    import pickle    
    global z1,StageII_intermediates,StageII_Pr,training_intermediates_ML
    global pzvidE,k,StageII_PZ #revised diag
    myFile = PREDICT_PAP + Threat_Rgn +".pkl"
    
    with open(myFile, 'rb') as pickle_file:
        logr = pickle.load(pickle_file)
 
    StageII_intermediates = BuildRegrTbl(StageII_Predictions)[:,0:-1] #omit last colum which is blank
    pzvidE = StageII_intermediates[:,0] #this is pzvid

    StageII_Pr = np.minimum(1,np.maximum(logr.predict_proba(StageII_intermediates[:,2:9]),0))

    pzInvty = pd.read_csv(PATH1 + "/pz_invtyStage1_2_Combo.csv",delimiter="|").set_index('PZidq')
    
    
    #Need the PZs of all StageII_
    #Will need this for minimbatch to add thse passegner to PZinvty4
    StageII_PZ= pzInvty[pzInvty["Probability"]=="NewB"]["PZ"]  #will need to revise

    qfile=PREDICT_PA2 + "/ViewComb1_NewB_"+ Threat_Rgn +"_.csv"

    with open(qfile,"w") as outFile:
        outFile.write("Id,Probability")

        for k in range(len(pzvidE)):

          v1=StageII_PZ[pzvidE[k]].replace("_z","_Zone")
          outFile.write("\n%s,%0.4f" % (v1,StageII_Pr[k][1] ) )


#########################################
#MAIN.........................
#########################################

''' 
pre-requisites. 
download files
get the pzvs from all the new files
  
Will need this for minimbatch to add thse passegner to PZinvty4 with prob="NewB

minibatch them and save as NewBatch rather than as Hidden
run the 48 metrics on the new files much as the hiddens, saving the 48max predicitons to a file

'''
    
PATH1 = r"C:/Users/Walzer/Documents/DS/Kaggle1/Homeland"     
EnvX = r"F:"  #C:
PPDF =  EnvX + r"/preProcessed"    #contains preprocessed .npy files 


Dm_dict = {"hip":256, "knee":200,"foot":200, "torso":225,"forearm":300,
               "upperarm":256, "calf":200, "groin":300,"upperchest":300,"myachingback":300,
               "uchback":300}

#revised 250 to 300

#verify. this is the number of StageII_ minibatch files
StageIILengthDict = {"hip":20, "knee":20,"foot":20, "torso":20,"forearm":20,
               "upperarm":19, "calf":20, "groin":10,"upperchest":2,"myachingback":2}


#tried hip at 260. now try 224
    
TRAIN_PATH = PATH1 + '/DNN/train/'  #Place to store the tensorboard logs
MODEL_PATH = PATH1+ '/DNN/model/' # Path where model files are stored   

PREDICT_PA2 = PATH1 + '/DNN/Predict'
PREDICT_PATH = PREDICT_PA2 + '/P_'
PREDICT_PAP = PATH1 + "/Pickled/Logistic_"

# dont exclude images even with poor quality if img.shape[0]>6 and img.shape[1]>6:
rz_dict = {"hip":[8,10], "knee":[11,12],"foot":[15,16], "torso":[6,7],"forearm":[2,4],
           "upperarm":[1,3], "calf":[13,14],   "groin":[9,-1],"upperchest":[5,-1],"myachingback":[17,-1],"uchback":[5,17]   }
#-1 signifies there is just one zone

#N_TRAIN_STEPS = 10       # The number of train steps (epochs) to run
    
rgnlist=["knee","hip","foot", "torso", "forearm","upperarm","calf","groin","upperchest","myachingback"]
    
rgnlist2=["hip","foot", "torso", "forearm","upperarm","calf","groin","upperchest","myachingback"]
    #-1 signifies there is just one zone


#TRAIN_TEST_SPLIT_RATIO = 0.2 #Ratio to split the FILE_LIST between train and test



prefix= '/NewB_' #        prefix= '/NewBatch_'

#these are zones not zm1
print("region list override")
for Threat_Rgn in rgnlist2:

    print ("region: {}\n".format(Threat_Rgn))
    #zONe1,zONe2 = rz_dict[Threat_Rgn][0], rz_dict[Threat_Rgn][1]

    #-1 signifies there is just one zone
    Dm1 = Dm_dict[Threat_Rgn]      #2nd dime height in pixels 
    
    #load the corresponding regional model
    ModelName = 'm_{}.tflearn'.format(Threat_Rgn)
    Learning_Rate = .01  
    model = None
    
    tf.reset_default_graph()
    
    with tf.Graph().as_default():
    ## Building deep neural network
    
        new_model = DNN(Dm1)
        #print("About to Load Model!")
        
        #new_model = tflearn.DNN(network, tensorboard_verbose=3)
        new_model.load(MODEL_PATH + ModelName,weights_only=True)
        model = new_model

    
        #predict Stage II
        #I had to break up StageII into sections of 1000 images to avoid crashing my PC
        for z in range(StageIILengthDict[Threat_Rgn]): #Fix
            print("\nModel Loaded.Predicting Stage II section:",z)
            StageII_pzid, dummy, StageII_im = input_pipeline(PPDF + prefix + Threat_Rgn + str(z) + ".npy",2) #hatch-lb is zeros
            StageII_im = StageII_im.reshape(-1, Dm1, Dm1, 1)
            StageII_Predictions_partial=model.predict({'imAges': StageII_im})
            StageII_Predictions_partial[:,0]=StageII_pzid #overwrite column zero
            np.savetxt(PREDICT_PA2 + "/NewB_"+ Threat_Rgn +str(z)+"_.csv", StageII_Predictions_partial, delimiter="|")
            
            if z==0:
                StageII_Predictions=StageII_Predictions_partial
            else:
                StageII_Predictions=np.concatenate( (StageII_Predictions, StageII_Predictions_partial ),axis=0)
                
        #Save the full prediciton set
        np.savetxt(PREDICT_PA2 + "/compositeNB_"+ Threat_Rgn +"_.csv", StageII_Predictions, delimiter="|")
        print("saved compsite Hidden single image pred file with shape ",StageII_Predictions.shape)

        viewCombiner2(Threat_Rgn) 






    
    
     
