<<<<<<< HEAD
#get passgrlabel2 needs df for code7

from __future__ import print_function, division
import numpy as np, os,re 
#import csv, time
#import cv2
import  pandas as pd
from matplotlib import pyplot as plt
#import scipy.stats as stats
#from scipy import ndimage as ndi

import tensorflow as tf #needed?
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.optimizers import Momentum


#################################
# DEFINED FUNCTIONS                       
################################
   


'''
gets the threat probabilities in a useful form (in TSAhelper)
Basic Descriptive Probabilities for Each Threat Zone
The labels provided in the contest treat the Passenger number and threat zone as a combined label.
But to calculate the descriptive stats I want to separate them so that we can get total counts for each threat zone.
Also, in this preprocessing approach, we will make individual examples out of each threat zone treating each threat zone as a separate model.
'''                



#Normalize and Zero Center.With the data cropped, we can normalize and zero center.
# work needs to be done to confirm a reasonable pixel mean for the zero center.
# normalize(image): Take segmented tsa image and normalize pixel values to be between 0 and 1
# parameters:      image - a tsa scan
# returns:         a normalized image
def normalize(image): #TSAhelper
    MIN_Bound = 0.0
    MAX_Bound = 255.0
    
    image = (image - MIN_Bound) / (MAX_Bound - MIN_Bound)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def plOT_image(img):
    #print("hello",img)
    plt.figure()
    plt.imshow(img, cmap='pink')
    plt.show()
    plt.close('all')


def get_train_test_file_list():
   
    global FILE_LIST, TRAIN_SET_FILE_LIST, TEST_SET_FILE_LIST

    if os.listdir(PPDF) == []:
        print ('No pre-processed data available.  Skipping ...')
        return
    
    FILE_LIST = [f for f in os.listdir(PPDF) if re.search(rgn_search, f)]
        
    lf=len(FILE_LIST) 
    tt_split = lf - max(int(lf*TRAIN_TEST_SPLIT_RATIO),1)
    
    TRAIN_SET_FILE_LIST = FILE_LIST[:tt_split]
    TEST_SET_FILE_LIST = FILE_LIST[tt_split:]
    
    print('Train/Test Split -> {} file(s) of {} used for testing'.format( 
          lf - tt_split, lf))

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
    minibatched_scans = np.load(os.path.join(PPDF, filename)) #Load a batch of pre-procd tz scans
    np.random.shuffle(minibatched_scans)     #Shuffle to randomize for input into the model
    
    # separate images and labels
    for example in minibatched_scans:
            pzid_batch.append(example[0])
            if param==1:
                label_batch.append(example[1]) #example[1] is a 2 item list
                image_batch.append(example[2])
            else:
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
    M1 = [int(.5* (Dm2 - x)) for x in ib2len]
    #max2_len = max(ib2len)
    
    ib3=np.zeros((len(iblen),Dm1,Dm2),dtype=np.float32)
    
    for jqz,scan2 in enumerate(ib2):
        if ib2len[jqz]<Dm2: #pad
            M2 = Dm2 - ib2len[jqz] - M1[jqz]
            scan3 = np.pad(scan2, pad_width= ((0, 0),(M1[jqz], M2)),mode= 'edge')
        elif ib2len[jqz]>Dm2: #crop
            M2 = -M1[jqz]+Dm2
            scan3= scan2[:, -M1[jqz]:M2]
        else:
            scan3=scan2
    
        ib3[jqz,:,:] =scan3

    pzid_batch = np.array(pzid_batch)
    if param==1:
        label_batch = np.asarray(label_batch, dtype=np.float32)
    else:
        label_batch=pzid_batch*0

    return pzid_batch, label_batch, ib3

def predict_sv(ID,Labz, Pred,regn,numb,suffix):
            Comb = np.concatenate( (ID.reshape(-1,1), Labz.reshape(-1,2)), axis=1)
            Comb = np.concatenate( (Comb,Pred ), axis=1)
            np.savetxt(PREDICT_PATH+regn+"_"+str(numb)+"_"+ suffix+".csv", Comb, delimiter="|" )
            #PZID, threat label (1-x,x) and prediction (1-x,x)


def train_II(rgn,number):

    for i in range(number): #multiple passes change to one pass??
    
        shuffle(TRAIN_SET_FILE_LIST) #shuffle list of files     
        
        # run through every batch in the training set. but keep validn set
        for ktr,each_batch in enumerate(TRAIN_SET_FILE_LIST):
         
            Batch_pzid, Batch_lb, Batch_im = input_pipeline(each_batch,1)
            Batch_im = Batch_im.reshape(-1, Dm1, Dm2, 1)
    
            # run the fit operation
            #n-epoch=1
            trainX, trainY =  {'imAges': Batch_im},  {'labels': Batch_lb}
            testX,testY = {'imAges': val_images} , {'labels': val_labels}
            # maybe n_epoch = orig val of N_TRAIN_STEPS
            #model.evaluate(trainX, trainY,batch_size=128 )
            model.fit(trainX, trainY , n_epoch=1, validation_set=(testX,testY), 
                      shuffle=True, snapshot_step=None, show_metric=True, 
                      run_id=ModelName)
    
            model.save(MODEL_PATH+"m_"+rgn+".tflearn")
            
            #save prediciton file on final pass
            if i == N_TRAIN_STEPS-1: # if one pass, modify or remove this check
                trainPred = model.predict(trainX)
                predict_sv(Batch_pzid, Batch_lb, trainPred,rgn,ktr,"TrainPred")
    
        #after the last batch, save validation set 
        testPred = model.predict(testX)
        predict_sv(val_pzids, val_labels, testPred,rgn,ktr,"TestPred")

         

def D1N(Dm1):
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





def BuildRegrTbl(param,XP):
    #builds regression table using the 3 highest scroing ML views and the 48 metrics
    global allPZID,prob48,p48 #diag
    col_dict = {"T":4, "H":1} #the training and validation sets have 5 columns, the Hidden have just 2

    allPZID  = [int(x+.5) for x in  sorted(set( XP[:,0])) ]
    Max3Table = np.zeros( (len(allPZID),10)   ) #new in 82. add column for 48Metric predictions
    Max3Table[:,0] = allPZID
    
    if param=="T":    
        p48 = pd.read_csv(PATH1 + "/predict48.csv",delimiter="|").set_index('pzvID')
        prob48 = p48["prediction"]
    elif param=="H":    
        p48H = pd.read_csv(PATH1 + "/predict48H.csv",delimiter="|").set_index('pzvID')
        prob48 = p48H["prediction"]


    for j,eachID in enumerate(allPZID):
       itemindex = np.where(XP[:,0]==eachID)[0]  #turns list of row indices

       if param=="T": #for training and test sets we pluck the label from column c1
           Max3Table[j,1]  =XP[itemindex[0],2] #retrieves the threat label  from XP and placesit in column 1

       pred = XP[itemindex,   col_dict[param] ] 
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


def viewCombiner(Threat_Rgn)  :
    import pickle    
    global z1,hidden_intermediates,Hidden_Pr,training_intermediates_ML
    global training_labels, training_intermediates_expanded #diag
    global slope, intercept, r_value, p_value, std_err
    
    #AllTrainingPredicts combines batches. columsn are pzvid, threat label (two columns) single image prob 2 cols. 
    #we only care about columns 0 2 and 4
    
    #z1 col1 is labels. 
    # 0 and 1 are allPZID and ?single image prediction
    # 234 567 are A B C  AB AC BC 
    #second to last column (8) is the 48 metrics prediction
    #last colis a placeholder for combined predictions
   
    z1 = BuildRegrTbl("T",AllTrainingPredicts) 
    
    training_intermediates_ML = z1[:,2:8] #PZID label maxview1 maxview2 maxview3 1x2 1x3 2x3 prediciton
    training_intermediates_expanded = z1[:,2:9] #PZID label maxview1 maxview2 maxview3 1x2 1x3 2x3 prediciton
    training_labels =  z1[:,1]
    
    from sklearn.linear_model import LogisticRegression as LR  
    logr = LR (penalty="l2",tol=.0001, class_weight=None,
          random_state=None, solver="liblinear",verbose=0,max_iter=200)
    print("\nfitting logistic regression..")

    logr.fit(training_intermediates_ML, training_labels)
    print("MaxView coefficients (ML only)",logr.coef_,"intercept",logr.intercept_,sep="|")
    max1_Predict = logr.predict_proba(training_intermediates_ML)[:,1]	#Probability estimates go in last column
    #can we get get rsquared directly from logr.
    r_value=np.corrcoef(max1_Predict, training_labels)[1,0]
    print ("R\u00b2 = {0:0.3f}\n".format(r_value**2) ) 
      
    logr.fit(training_intermediates_expanded, training_labels)
    print("\nincluding 48 metrics....\nMaxView coefficients" ,logr.coef_,
          "intercept", logr.intercept_, sep="|")
    max2_Predict = logr.predict_proba(training_intermediates_expanded)[:,1]	#Probability estimates go in last column


    with open(PREDICT_PAP +  Threat_Rgn +".pkl", 'wb') as outFile:  #fixed
        pickle.dump(logr, outFile)

    #can we get get rsquared directly from logr.
    r_value=np.corrcoef(max2_Predict, training_labels)[1,0]

    print ("R\u00b2 = {0:0.3f}\n".format(r_value**2) ) 
    z1[:,-1] =max2_Predict
    np.savetxt(PREDICT_PA2 + "/ViewComb1_Tr_"+ Threat_Rgn +"_.csv", z1, fmt='%d|%d|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f')     

    hidden_intermediates = BuildRegrTbl("H",HiddenPredictions)[:,0:-1] #omit last colum which is blank
    pzvidE = hidden_intermediates[:,0] #this is pzvid

    Hidden_Pr = np.minimum(1,np.maximum(logr.predict_proba(hidden_intermediates[:,2:9]),0))

    
    pzInvty = pd.read_csv(PATH1 + "/pz_invty4.csv",delimiter="|").set_index('PZidq')
    
    hidden= pzInvty[pzInvty["Probability"]=="Hidden"]["PZ"]

    qfile=PREDICT_PA2 + "/ViewComb1_Hd_"+ Threat_Rgn +"_.csv"
    with open(qfile,"w") as outFile:
        outFile.write("Id,Probability")

        for k in range(len(pzvidE)):
          v1=hidden[pzvidE[k]].replace("_z","_Zone")
          outFile.write("\n%s,%0.4f" % (v1,Hidden_Pr[k][1] ) )
 

#============================================================================+
#                                                                            |   
#                               M A I N                                      |
#                                                                            |   
#============================================================================+

### ______Main Routine _____________
#Seeking probability of contraband within a given threat zone. 
#ScanData = namedtuple('ScanData', ['header', 'data', 'real', 'imag', 'extension'])

global pzInvty
file_endings = [".aps",".ahi",".a3d",".a3daps"]
  
#envt= input("KOI or  MACK:   ").upper()
#if envt== "KOI":
envt="KOI"
PATH1 = r"C:/Users/Walzer/Documents/DS/Kaggle1/Homeland"     
EnvX = r"F:"  #C:

#elif envt=="MACK":
#    PATH1= r"C:/Users/Trapezoid LLC/Desktop/Homeland2"
#    EnvX= r"C:"
#else:
#    print("bad input")
  

INPUT_FOLD3R = EnvX + r"/_A3DAPS"     
PPDF =  EnvX + r"/preProcessed"    #contains preprocessed .npy files 
PPZV = EnvX + r"/_np_zvw2/" # EnvX + r"/_np_zvw2/"


# read labels into a dataframe
THREAT_LABELS =  PATH1 + r'/stage1_labels.csv'
Threats_df = pd.read_csv(THREAT_LABELS)  #labels csv file
Threats_df['Passenger'], Threats_df['Zone'] = Threats_df['Id'].str.split('_',1).str # Separate the zone and Passenger id 
Threats_df = Threats_df[['Passenger', 'Zone', 'Probability']]
Threats_df = Threats_df
#.set_index('Passenger')

   
# Dict to convert a 0 based threat zone index to the text we need to look up the label
ZONE_NAMES = {0: 'Zone1', 1: 'Zone2', 2: 'Zone3', 3: 'Zone4', 4: 'Zone5', 5: 'Zone6', 
              6: 'Zone7', 7: 'Zone8', 8: 'Zone9', 9: 'Zone10', 10: 'Zone11', 11: 'Zone12', 
              12: 'Zone13', 13: 'Zone14', 14: 'Zone15', 15: 'Zone16',
              16: 'Zone17'}



rgnlist=["calf","foot","forearm","groin","hip","knee","torso","upperarm",
         "upperchest","myachingback","uchback"]

my_vertices = pd.read_csv(PATH1+ r"/vertices_6.csv",sep="|")  
my_vertices.replace(np.nan,0)
#includes viewchooser
zone_lbl= [my_vertices['label'][x] for x in range(16)]
zone_lbl.append("upperback")


coefs = pd.read_csv(PATH1+ r"/phase_moons.csv",sep="|") 
#Fix ?specify datatypes of coefs are int in cols 2 thru
coefs2=coefs.set_index("L/R")
lCoef = coefs2.loc["lCoef"]
rCoef = coefs2.loc["rCoef"]

#list of passengers with data
df1 = pd.read_csv(PATH1+ r"/passgr_data_invty1.csv",sep="|")
#df2 = df1.loc[df1['small'] != 1]  #handrevised to process only the NEWER files
Pssgr_List1 = df1["passenger"].tolist() 

top_border  =0
ftyp= ".a3daps"

#we call first dim height and second width

FILE_LIST = []               # A list of the preProcesed .npy files to batch
TRAIN_SET_FILE_LIST = []     #The list of .npy files to be used for training
TEST_SET_FILE_LIST = []      #The list of .npy files to be used for testing
TRAIN_TEST_SPLIT_RATIO = 0.2 #Ratio to split the FILE_LIST between train and test
TRAIN_PATH = PATH1 + '/DNN2/train/'  #Place to store the tensorboard logs
MODEL_PATH = PATH1+ '/DNN2/model/' # Path where model files are stored   
MODEL_PATHold = PATH1+ '/DNN/model/' # Path where model files are stored 


PREDICT_PA2 = PATH1 + '/DNN2/Predict'
PREDICT_PATH = PREDICT_PA2 + '/P_'
PREDICT_PAP = PATH1 + "/Pickled/Logistic_"

#Tuneable Parameters
Learning_Rate = .01  #he had -3    
 
N_TRAIN_STEPS = int(input("# of training steps:" ))       # The number of train steps (epochs) to run



# dont exclude images even with poor quality if img.shape[0]>6 and img.shape[1]>6:
rz_dict = {"hip":[8,10], "knee":[11,12],"foot":[15,16], "torso":[6,7],"forearm":[2,4],
           "upperarm":[1,3], "calf":[13,14],   "groin":[9,-1],"upperchest":[5,-1],"myachingback":[17,-1],"uchback":[5,17]   }
#-1 signifies there is just one zone

Dm_dict = {"hip":256, "knee":200,"foot":200, "torso":225,"forearm":300,
           "upperarm":256, "calf":200, "groin":200,"upperchest":300,"myachingback":300,
           "uchback":300}
#verify the uppearm


print(rgnlist)
Threat_Rgn="" 
while Threat_Rgn not in rgnlist: 
    Threat_Rgn = input("region : ").lower()

print("=========================================================")
print ("- - - " + Threat_Rgn + " - - - -     passes=",N_TRAIN_STEPS)
print("=========================================================")
rgn_search =re.compile( Threat_Rgn + '-b')


Dm1 = Dm_dict[Threat_Rgn]      #2nd dime height in pixels 
Dm2 = Dm1    #it likes squares
   

#-1 signifies there is just one zone
Dm1 = Dm_dict[Threat_Rgn]      #2nd dime height in pixels 

#load the corresponding regional model
ModelName = 'm_{}.tflearn'.format(Threat_Rgn)

model = None

tf.reset_default_graph()

with tf.Graph().as_default():
## Building deep neural network

    new_model = D1N(Dm1)
    #print("About to Load Model!")
    
    #new_model = tflearn.DNN(network, tensorboard_verbose=3)
    new_model.load(MODEL_PATHold + ModelName,weights_only=True)
    model = new_model
    print("\nModelLoaded.")


    rgn=Threat_Rgn
    #TFLearn treats each "minibatch" as an epoch.   
    global eachfile, AllTrainingPredicts, HiddenPredictions, Hid_pzid
 


    get_train_test_file_list()  # get train and test batches##returns TRAIN_SET_FILE_LIST as a global
    #defines FILE_LIST, TrainingFiles, TEST_SET_FILE_LIST
    #tf.reset_default_graph()
    #model = DNN(Dm1, Dm2, Learning_Rate)    # instantiate model
    

    
    #input_pipeline    
    print ('Train Set -----------------------------')
    
    for f_in in TRAIN_SET_FILE_LIST:
        pzid_batch,  label_batch, image_batch = input_pipeline(f_in,1)
        print (' -> images shape {}x{}x{}'.format(len(image_batch),
               len(image_batch[0]), len(image_batch[0][0])),end="; ")
        print ('  labels shape   {}x{}'.format(len(label_batch), len(label_batch[0])))
        
    print ('Test Set -----------------------------')
    
    for f_in in TEST_SET_FILE_LIST:
        pzid_batch,  label_batch, image_batch = input_pipeline(f_in,1)
        print (' -> images shape {}x{}x{}'.format(len(image_batch), 
                                                    len(image_batch[0]), 
                                                    len(image_batch[0][0])),end="; ")
        print ('  labels shape   {}x{}'.format(len(label_batch), len(label_batch[0])))
    
    
    #i think he utilizes this rather than incr epochs so he can shuffle more frequently
    #Fix get program name automaticallly
    #MODEL_NAME = ('{}x-tz-{}'.format('v71', Dm1, Threat_Rgn ))
    
    shuffle(TRAIN_SET_FILE_LIST)
    print ('After Shuffling ->', TRAIN_SET_FILE_LIST)
    
    #trainer(Threat_Rgn)  #creates  global var AllTrainingPredicts (#of Views, 5) PZID ThreatLabel MaxView1 MaxView2 MaxView3 

    val_images,    val_labels = [],[]
    # read in the validation test set
    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_pzids, val_labels, val_images  = input_pipeline(test_f_in,1)
        else:
            tmp_pzid_batch, tmp_label_batch, tmp_image_batch,  = input_pipeline(test_f_in,1)
            
            val_pzids  = np.concatenate((tmp_pzid_batch,  val_pzids),  axis=0) 
            val_images = np.concatenate((tmp_image_batch, val_images), axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)

        val_images = val_images.reshape(-1, Dm1, Dm2, 1)
    
  
    train_II(rgn,N_TRAIN_STEPS)

    #assembles the prediciton files into an integrted set. check
    flag=0
    for eachfile in os.listdir(PREDICT_PA2):
       
       if rgn in eachfile and eachfile.split('.')[-1] =="csv" and "composite" not in eachfile and "max" not in eachfile and "ViewComb" not in eachfile:
           b=np.loadtxt(PREDICT_PA2+ r"/" + eachfile,delimiter="|")
           #print(x,b.shape)
           try:
               if b.shape[1]==5:
                   if flag==0:
                       AllTrainingPredicts=b
                       flag=1
                   else:    
                       AllTrainingPredicts=np.concatenate( (AllTrainingPredicts,b ),axis=0)
               else:
                   print(eachfile,"u1",b.shape)
           except:
                   print(eachfile,"u2",b.shape)
    
    
    np.savetxt(PREDICT_PA2 + "/composite_"+ rgn +"_.csv", AllTrainingPredicts, delimiter="|")
    print("saved compsite training single image pred file with shape ",AllTrainingPredicts.shape)

    Hid_pzid, dummy, Hid_im = input_pipeline(PPDF + '/Hidden_' + rgn+".npy",2) #hatch-lb is zeros
    Hid_im = Hid_im.reshape(-1, Dm1, Dm2, 1)
    HiddenPredictions=model.predict({'imAges': Hid_im})
    HiddenPredictions[:,0]=Hid_pzid #overwrite column zero

    np.savetxt(PREDICT_PA2 + "/compositeH_"+ rgn +"_.csv", HiddenPredictions, delimiter="|")
    print("saved compsite Hidden single image pred file with shape ",HiddenPredictions.shape)  


    
    viewCombiner(Threat_Rgn)

=======
#get passgrlabel2 needs df for code7

from __future__ import print_function, division
import numpy as np, os,re 
#import csv, time
#import cv2
import  pandas as pd
from matplotlib import pyplot as plt
#import scipy.stats as stats
#from scipy import ndimage as ndi

import tensorflow as tf #needed?
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.optimizers import Momentum


#################################
# DEFINED FUNCTIONS                       
################################
   


'''
gets the threat probabilities in a useful form (in TSAhelper)
Basic Descriptive Probabilities for Each Threat Zone
The labels provided in the contest treat the Passenger number and threat zone as a combined label.
But to calculate the descriptive stats I want to separate them so that we can get total counts for each threat zone.
Also, in this preprocessing approach, we will make individual examples out of each threat zone treating each threat zone as a separate model.
'''                



#Normalize and Zero Center.With the data cropped, we can normalize and zero center.
# work needs to be done to confirm a reasonable pixel mean for the zero center.
# normalize(image): Take segmented tsa image and normalize pixel values to be between 0 and 1
# parameters:      image - a tsa scan
# returns:         a normalized image
def normalize(image): #TSAhelper
    MIN_Bound = 0.0
    MAX_Bound = 255.0
    
    image = (image - MIN_Bound) / (MAX_Bound - MIN_Bound)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


def plOT_image(img):
    #print("hello",img)
    plt.figure()
    plt.imshow(img, cmap='pink')
    plt.show()
    plt.close('all')


def get_train_test_file_list():
   
    global FILE_LIST, TRAIN_SET_FILE_LIST, TEST_SET_FILE_LIST

    if os.listdir(PPDF) == []:
        print ('No pre-processed data available.  Skipping ...')
        return
    
    FILE_LIST = [f for f in os.listdir(PPDF) if re.search(rgn_search, f)]
        
    lf=len(FILE_LIST) 
    tt_split = lf - max(int(lf*TRAIN_TEST_SPLIT_RATIO),1)
    
    TRAIN_SET_FILE_LIST = FILE_LIST[:tt_split]
    TEST_SET_FILE_LIST = FILE_LIST[tt_split:]
    
    print('Train/Test Split -> {} file(s) of {} used for testing'.format( 
          lf - tt_split, lf))

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
    minibatched_scans = np.load(os.path.join(PPDF, filename)) #Load a batch of pre-procd tz scans
    np.random.shuffle(minibatched_scans)     #Shuffle to randomize for input into the model
    
    # separate images and labels
    for example in minibatched_scans:
            pzid_batch.append(example[0])
            if param==1:
                label_batch.append(example[1]) #example[1] is a 2 item list
                image_batch.append(example[2])
            else:
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
    M1 = [int(.5* (Dm2 - x)) for x in ib2len]
    #max2_len = max(ib2len)
    
    ib3=np.zeros((len(iblen),Dm1,Dm2),dtype=np.float32)
    
    for jqz,scan2 in enumerate(ib2):
        if ib2len[jqz]<Dm2: #pad
            M2 = Dm2 - ib2len[jqz] - M1[jqz]
            scan3 = np.pad(scan2, pad_width= ((0, 0),(M1[jqz], M2)),mode= 'edge')
        elif ib2len[jqz]>Dm2: #crop
            M2 = -M1[jqz]+Dm2
            scan3= scan2[:, -M1[jqz]:M2]
        else:
            scan3=scan2
    
        ib3[jqz,:,:] =scan3

    pzid_batch = np.array(pzid_batch)
    if param==1:
        label_batch = np.asarray(label_batch, dtype=np.float32)
    else:
        label_batch=pzid_batch*0

    return pzid_batch, label_batch, ib3

def predict_sv(ID,Labz, Pred,regn,numb,suffix):
            Comb = np.concatenate( (ID.reshape(-1,1), Labz.reshape(-1,2)), axis=1)
            Comb = np.concatenate( (Comb,Pred ), axis=1)
            np.savetxt(PREDICT_PATH+regn+"_"+str(numb)+"_"+ suffix+".csv", Comb, delimiter="|" )
            #PZID, threat label (1-x,x) and prediction (1-x,x)


def train_II(rgn,number):

    for i in range(number): #multiple passes change to one pass??
    
        shuffle(TRAIN_SET_FILE_LIST) #shuffle list of files     
        
        # run through every batch in the training set. but keep validn set
        for ktr,each_batch in enumerate(TRAIN_SET_FILE_LIST):
         
            Batch_pzid, Batch_lb, Batch_im = input_pipeline(each_batch,1)
            Batch_im = Batch_im.reshape(-1, Dm1, Dm2, 1)
    
            # run the fit operation
            #n-epoch=1
            trainX, trainY =  {'imAges': Batch_im},  {'labels': Batch_lb}
            testX,testY = {'imAges': val_images} , {'labels': val_labels}
            # maybe n_epoch = orig val of N_TRAIN_STEPS
            #model.evaluate(trainX, trainY,batch_size=128 )
            model.fit(trainX, trainY , n_epoch=1, validation_set=(testX,testY), 
                      shuffle=True, snapshot_step=None, show_metric=True, 
                      run_id=ModelName)
    
            model.save(MODEL_PATH+"m_"+rgn+".tflearn")
            
            #save prediciton file on final pass
            if i == N_TRAIN_STEPS-1: # if one pass, modify or remove this check
                trainPred = model.predict(trainX)
                predict_sv(Batch_pzid, Batch_lb, trainPred,rgn,ktr,"TrainPred")
    
        #after the last batch, save validation set 
        testPred = model.predict(testX)
        predict_sv(val_pzids, val_labels, testPred,rgn,ktr,"TestPred")

         

def D1N(Dm1):
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





def BuildRegrTbl(param,XP):
    #builds regression table using the 3 highest scroing ML views and the 48 metrics
    global allPZID,prob48,p48 #diag
    col_dict = {"T":4, "H":1} #the training and validation sets have 5 columns, the Hidden have just 2

    allPZID  = [int(x+.5) for x in  sorted(set( XP[:,0])) ]
    Max3Table = np.zeros( (len(allPZID),10)   ) #new in 82. add column for 48Metric predictions
    Max3Table[:,0] = allPZID
    
    if param=="T":    
        p48 = pd.read_csv(PATH1 + "/predict48.csv",delimiter="|").set_index('pzvID')
        prob48 = p48["prediction"]
    elif param=="H":    
        p48H = pd.read_csv(PATH1 + "/predict48H.csv",delimiter="|").set_index('pzvID')
        prob48 = p48H["prediction"]


    for j,eachID in enumerate(allPZID):
       itemindex = np.where(XP[:,0]==eachID)[0]  #turns list of row indices

       if param=="T": #for training and test sets we pluck the label from column c1
           Max3Table[j,1]  =XP[itemindex[0],2] #retrieves the threat label  from XP and placesit in column 1

       pred = XP[itemindex,   col_dict[param] ] 
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


def viewCombiner(Threat_Rgn)  :
    import pickle    
    global z1,hidden_intermediates,Hidden_Pr,training_intermediates_ML
    global training_labels, training_intermediates_expanded #diag
    global slope, intercept, r_value, p_value, std_err
    
    #AllTrainingPredicts combines batches. columsn are pzvid, threat label (two columns) single image prob 2 cols. 
    #we only care about columns 0 2 and 4
    
    #z1 col1 is labels. 
    # 0 and 1 are allPZID and ?single image prediction
    # 234 567 are A B C  AB AC BC 
    #second to last column (8) is the 48 metrics prediction
    #last colis a placeholder for combined predictions
   
    z1 = BuildRegrTbl("T",AllTrainingPredicts) 
    
    training_intermediates_ML = z1[:,2:8] #PZID label maxview1 maxview2 maxview3 1x2 1x3 2x3 prediciton
    training_intermediates_expanded = z1[:,2:9] #PZID label maxview1 maxview2 maxview3 1x2 1x3 2x3 prediciton
    training_labels =  z1[:,1]
    
    from sklearn.linear_model import LogisticRegression as LR  
    logr = LR (penalty="l2",tol=.0001, class_weight=None,
          random_state=None, solver="liblinear",verbose=0,max_iter=200)
    print("\nfitting logistic regression..")

    logr.fit(training_intermediates_ML, training_labels)
    print("MaxView coefficients (ML only)",logr.coef_,"intercept",logr.intercept_,sep="|")
    max1_Predict = logr.predict_proba(training_intermediates_ML)[:,1]	#Probability estimates go in last column
    #can we get get rsquared directly from logr.
    r_value=np.corrcoef(max1_Predict, training_labels)[1,0]
    print ("R\u00b2 = {0:0.3f}\n".format(r_value**2) ) 
      
    logr.fit(training_intermediates_expanded, training_labels)
    print("\nincluding 48 metrics....\nMaxView coefficients" ,logr.coef_,
          "intercept", logr.intercept_, sep="|")
    max2_Predict = logr.predict_proba(training_intermediates_expanded)[:,1]	#Probability estimates go in last column


    with open(PREDICT_PAP +  Threat_Rgn +".pkl", 'wb') as outFile:  #fixed
        pickle.dump(logr, outFile)

    #can we get get rsquared directly from logr.
    r_value=np.corrcoef(max2_Predict, training_labels)[1,0]

    print ("R\u00b2 = {0:0.3f}\n".format(r_value**2) ) 
    z1[:,-1] =max2_Predict
    np.savetxt(PREDICT_PA2 + "/ViewComb1_Tr_"+ Threat_Rgn +"_.csv", z1, fmt='%d|%d|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f')     

    hidden_intermediates = BuildRegrTbl("H",HiddenPredictions)[:,0:-1] #omit last colum which is blank
    pzvidE = hidden_intermediates[:,0] #this is pzvid

    Hidden_Pr = np.minimum(1,np.maximum(logr.predict_proba(hidden_intermediates[:,2:9]),0))

    
    pzInvty = pd.read_csv(PATH1 + "/pz_invty4.csv",delimiter="|").set_index('PZidq')
    
    hidden= pzInvty[pzInvty["Probability"]=="Hidden"]["PZ"]

    qfile=PREDICT_PA2 + "/ViewComb1_Hd_"+ Threat_Rgn +"_.csv"
    with open(qfile,"w") as outFile:
        outFile.write("Id,Probability")

        for k in range(len(pzvidE)):
          v1=hidden[pzvidE[k]].replace("_z","_Zone")
          outFile.write("\n%s,%0.4f" % (v1,Hidden_Pr[k][1] ) )
 

#============================================================================+
#                                                                            |   
#                               M A I N                                      |
#                                                                            |   
#============================================================================+

### ______Main Routine _____________
#Seeking probability of contraband within a given threat zone. 
#ScanData = namedtuple('ScanData', ['header', 'data', 'real', 'imag', 'extension'])

global pzInvty
file_endings = [".aps",".ahi",".a3d",".a3daps"]
  
#envt= input("KOI or  MACK:   ").upper()
#if envt== "KOI":
envt="KOI"
PATH1 = r"C:/Users/Walzer/Documents/DS/Kaggle1/Homeland"     
EnvX = r"F:"  #C:

#elif envt=="MACK":
#    PATH1= r"C:/Users/Trapezoid LLC/Desktop/Homeland2"
#    EnvX= r"C:"
#else:
#    print("bad input")
  

INPUT_FOLD3R = EnvX + r"/_A3DAPS"     
PPDF =  EnvX + r"/preProcessed"    #contains preprocessed .npy files 
PPZV = EnvX + r"/_np_zvw2/" # EnvX + r"/_np_zvw2/"


# read labels into a dataframe
THREAT_LABELS =  PATH1 + r'/stage1_labels.csv'
Threats_df = pd.read_csv(THREAT_LABELS)  #labels csv file
Threats_df['Passenger'], Threats_df['Zone'] = Threats_df['Id'].str.split('_',1).str # Separate the zone and Passenger id 
Threats_df = Threats_df[['Passenger', 'Zone', 'Probability']]
Threats_df = Threats_df
#.set_index('Passenger')

   
# Dict to convert a 0 based threat zone index to the text we need to look up the label
ZONE_NAMES = {0: 'Zone1', 1: 'Zone2', 2: 'Zone3', 3: 'Zone4', 4: 'Zone5', 5: 'Zone6', 
              6: 'Zone7', 7: 'Zone8', 8: 'Zone9', 9: 'Zone10', 10: 'Zone11', 11: 'Zone12', 
              12: 'Zone13', 13: 'Zone14', 14: 'Zone15', 15: 'Zone16',
              16: 'Zone17'}



rgnlist=["calf","foot","forearm","groin","hip","knee","torso","upperarm",
         "upperchest","myachingback","uchback"]

my_vertices = pd.read_csv(PATH1+ r"/vertices_6.csv",sep="|")  
my_vertices.replace(np.nan,0)
#includes viewchooser
zone_lbl= [my_vertices['label'][x] for x in range(16)]
zone_lbl.append("upperback")


coefs = pd.read_csv(PATH1+ r"/phase_moons.csv",sep="|") 
#Fix ?specify datatypes of coefs are int in cols 2 thru
coefs2=coefs.set_index("L/R")
lCoef = coefs2.loc["lCoef"]
rCoef = coefs2.loc["rCoef"]

#list of passengers with data
df1 = pd.read_csv(PATH1+ r"/passgr_data_invty1.csv",sep="|")
#df2 = df1.loc[df1['small'] != 1]  #handrevised to process only the NEWER files
Pssgr_List1 = df1["passenger"].tolist() 

top_border  =0
ftyp= ".a3daps"

#we call first dim height and second width

FILE_LIST = []               # A list of the preProcesed .npy files to batch
TRAIN_SET_FILE_LIST = []     #The list of .npy files to be used for training
TEST_SET_FILE_LIST = []      #The list of .npy files to be used for testing
TRAIN_TEST_SPLIT_RATIO = 0.2 #Ratio to split the FILE_LIST between train and test
TRAIN_PATH = PATH1 + '/DNN2/train/'  #Place to store the tensorboard logs
MODEL_PATH = PATH1+ '/DNN2/model/' # Path where model files are stored   
MODEL_PATHold = PATH1+ '/DNN/model/' # Path where model files are stored 


PREDICT_PA2 = PATH1 + '/DNN2/Predict'
PREDICT_PATH = PREDICT_PA2 + '/P_'
PREDICT_PAP = PATH1 + "/Pickled/Logistic_"

#Tuneable Parameters
Learning_Rate = .01  #he had -3    
 
N_TRAIN_STEPS = int(input("# of training steps:" ))       # The number of train steps (epochs) to run



# dont exclude images even with poor quality if img.shape[0]>6 and img.shape[1]>6:
rz_dict = {"hip":[8,10], "knee":[11,12],"foot":[15,16], "torso":[6,7],"forearm":[2,4],
           "upperarm":[1,3], "calf":[13,14],   "groin":[9,-1],"upperchest":[5,-1],"myachingback":[17,-1],"uchback":[5,17]   }
#-1 signifies there is just one zone

Dm_dict = {"hip":256, "knee":200,"foot":200, "torso":225,"forearm":300,
           "upperarm":256, "calf":200, "groin":200,"upperchest":300,"myachingback":300,
           "uchback":300}
#verify the uppearm


print(rgnlist)
Threat_Rgn="" 
while Threat_Rgn not in rgnlist: 
    Threat_Rgn = input("region : ").lower()

print("=========================================================")
print ("- - - " + Threat_Rgn + " - - - -     passes=",N_TRAIN_STEPS)
print("=========================================================")
rgn_search =re.compile( Threat_Rgn + '-b')


Dm1 = Dm_dict[Threat_Rgn]      #2nd dime height in pixels 
Dm2 = Dm1    #it likes squares
   

#-1 signifies there is just one zone
Dm1 = Dm_dict[Threat_Rgn]      #2nd dime height in pixels 

#load the corresponding regional model
ModelName = 'm_{}.tflearn'.format(Threat_Rgn)

model = None

tf.reset_default_graph()

with tf.Graph().as_default():
## Building deep neural network

    new_model = D1N(Dm1)
    #print("About to Load Model!")
    
    #new_model = tflearn.DNN(network, tensorboard_verbose=3)
    new_model.load(MODEL_PATHold + ModelName,weights_only=True)
    model = new_model
    print("\nModelLoaded.")


    rgn=Threat_Rgn
    #TFLearn treats each "minibatch" as an epoch.   
    global eachfile, AllTrainingPredicts, HiddenPredictions, Hid_pzid
 


    get_train_test_file_list()  # get train and test batches##returns TRAIN_SET_FILE_LIST as a global
    #defines FILE_LIST, TrainingFiles, TEST_SET_FILE_LIST
    #tf.reset_default_graph()
    #model = DNN(Dm1, Dm2, Learning_Rate)    # instantiate model
    

    
    #input_pipeline    
    print ('Train Set -----------------------------')
    
    for f_in in TRAIN_SET_FILE_LIST:
        pzid_batch,  label_batch, image_batch = input_pipeline(f_in,1)
        print (' -> images shape {}x{}x{}'.format(len(image_batch),
               len(image_batch[0]), len(image_batch[0][0])),end="; ")
        print ('  labels shape   {}x{}'.format(len(label_batch), len(label_batch[0])))
        
    print ('Test Set -----------------------------')
    
    for f_in in TEST_SET_FILE_LIST:
        pzid_batch,  label_batch, image_batch = input_pipeline(f_in,1)
        print (' -> images shape {}x{}x{}'.format(len(image_batch), 
                                                    len(image_batch[0]), 
                                                    len(image_batch[0][0])),end="; ")
        print ('  labels shape   {}x{}'.format(len(label_batch), len(label_batch[0])))
    
    
    #i think he utilizes this rather than incr epochs so he can shuffle more frequently
    #Fix get program name automaticallly
    #MODEL_NAME = ('{}x-tz-{}'.format('v71', Dm1, Threat_Rgn ))
    
    shuffle(TRAIN_SET_FILE_LIST)
    print ('After Shuffling ->', TRAIN_SET_FILE_LIST)
    
    #trainer(Threat_Rgn)  #creates  global var AllTrainingPredicts (#of Views, 5) PZID ThreatLabel MaxView1 MaxView2 MaxView3 

    val_images,    val_labels = [],[]
    # read in the validation test set
    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_pzids, val_labels, val_images  = input_pipeline(test_f_in,1)
        else:
            tmp_pzid_batch, tmp_label_batch, tmp_image_batch,  = input_pipeline(test_f_in,1)
            
            val_pzids  = np.concatenate((tmp_pzid_batch,  val_pzids),  axis=0) 
            val_images = np.concatenate((tmp_image_batch, val_images), axis=0)
            val_labels = np.concatenate((tmp_label_batch, val_labels), axis=0)

        val_images = val_images.reshape(-1, Dm1, Dm2, 1)
    
  
    train_II(rgn,N_TRAIN_STEPS)

    #assembles the prediciton files into an integrted set. check
    flag=0
    for eachfile in os.listdir(PREDICT_PA2):
       
       if rgn in eachfile and eachfile.split('.')[-1] =="csv" and "composite" not in eachfile and "max" not in eachfile and "ViewComb" not in eachfile:
           b=np.loadtxt(PREDICT_PA2+ r"/" + eachfile,delimiter="|")
           #print(x,b.shape)
           try:
               if b.shape[1]==5:
                   if flag==0:
                       AllTrainingPredicts=b
                       flag=1
                   else:    
                       AllTrainingPredicts=np.concatenate( (AllTrainingPredicts,b ),axis=0)
               else:
                   print(eachfile,"u1",b.shape)
           except:
                   print(eachfile,"u2",b.shape)
    
    
    np.savetxt(PREDICT_PA2 + "/composite_"+ rgn +"_.csv", AllTrainingPredicts, delimiter="|")
    print("saved compsite training single image pred file with shape ",AllTrainingPredicts.shape)

    Hid_pzid, dummy, Hid_im = input_pipeline(PPDF + '/Hidden_' + rgn+".npy",2) #hatch-lb is zeros
    Hid_im = Hid_im.reshape(-1, Dm1, Dm2, 1)
    HiddenPredictions=model.predict({'imAges': Hid_im})
    HiddenPredictions[:,0]=Hid_pzid #overwrite column zero

    np.savetxt(PREDICT_PA2 + "/compositeH_"+ rgn +"_.csv", HiddenPredictions, delimiter="|")
    print("saved compsite Hidden single image pred file with shape ",HiddenPredictions.shape)  


    
    viewCombiner(Threat_Rgn)

>>>>>>> 22f1a8e454425c2160add7f14e9f2915e5913a6e
