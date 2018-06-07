'''
this version from first round of Kaggle Contest includes some preprocessing functions
excluded from version 103
a previous version had forearms to 210. 
vertices_5 eliminates arm views 16 and 48
'''

from __future__ import print_function, division
import numpy as np, os, csv, time, re
import cv2, pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
from scipy import ndimage as ndi
import warnings

#################################
# DEFINED FUNCTIONS                       
################################

#takes an aps file and creates a dict of the data, returns all of the fields in the header.  (Kaggle)
#read image header (first 512 bytes)
def read_header(infile):       
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['fidef le_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
    return h


#  reads and rescales any of the four image types
# infile:             an .aps, .aps3d, .a3d, or ahi file
# returns:            the stack of images
# note:               word_type == 7 is an np.float32, word_type == 4 is np.uint16      
def read_data(infile):
    
    # read in header and get dimensions
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])

    extension = os.path.splitext(infile)[1]
    
    with open(infile, 'rb') as fid:
          
        # skip the header
        fid.seek(512) 

        # handle .aps and .a3aps files
        if extension == '.aps' or extension == '.a3daps':
        
            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

            elif(h['word_type']==4): 
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor'] 
            data = data.reshape(nx, ny, nt, order='F').copy()

        # handle .a3d files
        elif extension == '.a3d':
              
            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
                
            elif(h['word_type']==4):
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, nt, ny, order='F').copy() 
            
        # handle .ahi files
        elif extension == '.ahi':
            data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
            data = data.reshape(2, ny, nx, nt, order='F').copy()
            real = data[0,:,:,:].copy()
            imag = data[1,:,:,:].copy()

        if extension != '.ahi':
            return data
        else:
            return real, imag



'''
gets the threat probabilities in a useful form (in TSAhelper)
Basic Descriptive Probabilities for Each Threat Zone
The labels provided in the contest treat the Passenger number and threat zone as a combined label.
But to calculate the descriptive stats I want to separate them so that we can get total counts for each threat zone.
Also, in this preprocessing approach, we will make individual examples out of each threat zone treating each threat zone as a separate model.
'''                
def get_hit_rate_stats(infile): #labels csv file
    # pull the labels for a given patient
    df = pd.read_csv(infile)

    # Separate the zone and patient id into a df
    df['Passenger'], df['Zone'] = df['Id'].str.split('_',1).str
    df = df[['Passenger', 'Zone', 'Probability']]

    # make a df of the sums and counts by zone and calculate hit rate per zone, then sort high to low
    df_summary = df.groupby('Zone')['Probability'].agg(['sum','count'])
    df_summary['Zone'] = df_summary.index
    df_summary['pct'] = df_summary['sum'] / df_summary['count']
    df_summary.sort_values('pct', axis=0, ascending= False, inplace=True)
    
    return df_summary #a dataframe of the summary hit probabilities


#Passenger Threat List
# zone_num:                                a 0 based threat zone index
# df:                                      a df like that returned from get_Pssgr_labels(...)
# returns:                                 [0,1] if contraband is present, [1,0] if it isnt



# returns the nth image from the image stack (TSA HELPER)
def get_single_image(aps_file, nth_slice): 
    img = read_data(aps_file)  # comes in as shape(512, 620, 16)
    img = img.transpose() #    shape(16, 620, 512) first diemsinon is slice
    return np.flipud(img[nth_slice]) #returns the nth image from image stack




#converts a ATI scan to grayscale. (in TSAhelper)
#note: Most image preprocessing functions want the image as grayscale
def convert_to_grayscale(aps_img):
    # scale pixel values to grayscale
    base_range = np.amax(aps_img) - np.amin(aps_img)
    rescaled_range = 255 - 0
    img_rescaled = (((aps_img - np.amin(aps_img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled) #greyscale image

'''
Spreading the Spectrum
most pixels are found between a value of 0 and and about 25. The entire range of grayscale values in the scan is less than ~125.
You can also see a fair amount of ghosting or noise around the core image.
Maybe the millimeter wave technology scatters some noise?  Not sure... Anyway, if someone knows what this is caused by, drop a note in the comments.

1) threshold the background. I've played quite a bit with the threshmin setting (12 has worked best so far), but this is obviously a parameter to play with.
2) equalize the distribution of the grayscale spectrum in this image. See this tutorial if you want to learn more about this technique.redistributes pixel values to a the full grayscale spectrum in order to increase contrast.
'''
#     applies a histogram equalization transformation  #TSAHELPER
def spread_spectrum(img):
    img = stats.threshold(img, threshmin=12, newval=0)
    
    # see http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img= clahe.apply(img)
    
    return img #transformed scan

'''
Masking the Region of Interest("ROI") (TSAHelper)
Using the slice lists from above, getting a set of masked images for a given threat zone is straight forward.
The same note applies here as in the 4x4 visualization above, I used a cv2.resize to get around a size constraint (see above), therefore the images are quite blurry at this resolution.
Note that the blurriness only applies to the unit test and visualization.
The data returned by this function is at full resolution.
'''



# img:                             the image to be masked
# vertices:                        a set of vertices that define the region of interest
def roi(img, vertices): #uses vertices to mask image
  
    mask = np.zeros_like(img)     # blank mask
    cv2.fillPoly(mask, [vertices], 255)     # fill the mask
    # now only show the area that is the mask
    return cv2.bitwise_and(img, mask) #masked image

'''
Cropping the Images ()
The same note applies here as in the 4x4 visualization above, 
I used a cv2.resize to get around a size constraint (see above),
 therefore the images are quite blurry at this resolution.
If you do not face this size constraint, drop the resize.
Note that the blurriness only applies the unit test and visualization.
The data returned by this function is at full resolution.
'''


# crop_list: =        a crop_list entry with [x , y, width, height]
def crop(img, crop_list): #TSAhelper)

    x_coord = crop_list[0]
    y_coord = crop_list[1]
    width = crop_list[2]
    height = crop_list[3]
    cropped_img = img[x_coord:x_coord+width, y_coord:y_coord+height]
    
    return cropped_img #cropped image

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


def top_finder(image): #660x512 numpy
    #dark rows (highest value<.125) 
    top_border =  np.argmax(np.amax(image,1)>.125)   #find first light row.  dark row (dark means no value>.125)
    #zz = np.amax(image,0)>.125 #light columns
    return top_border

def plOT_image(img):
    #print("hello",img)
    plt.figure()
    plt.imshow(img, cmap=COLORMAP)
    plt.show()
    plt.close('all')



#saves selected images to output file  
def sav2file(an_img,d1,d2,xlabel,ViewName,psgrid):
    #fig, axarr = plt.subplots(nRows=1, nCols=2, figsize=(d1, d2))
    plt.figure(figsize=(d1, d2))
    plt.imshow(an_img, cmap=COLORMAP) #axarr[0].imshow(an_img, cmap=COLORMAP)
    #plt.subplot(122)
    #plt.hist(an_img.flatten(), bins=256, color='c')
    #plt.xlabel(xlabel)
    #plt.ylabel("Frequency")
    outfile= OUTPUT_DATA_FOLDER +"/" + psgrid + "/" +  ViewName+'.png'
    plt.savefig(outfile, bbox_inches='tight')
    #plt.close(figure)
    plt.close()
    print(outfile, " saved_____________\n")
   

def zoneslicelist1(zone,view,filetype):
    global zcl #verify
    import math
    zon=zone     #zone 0 thru 16 correspond to TSA threatzones 1 to 17
    viewq=view*vMult[filetype] 
    if zon==16:
        zon=4
    #some views are not worth saving for some threat zones    
    if zon==4 and 16<=viewq<=48:
        zsl=nullZSL
    elif zone==16 and (viewq<16 or viewq>48):
        zsl=nullZSL
    else:

        cos1=math.cos(viewq/64*2*math.pi )    
        width_adj = my_vertices['width'][zon]/2*lCoef[str(viewq)] 
        
        left2 =  (my_vertices['left'][zon]-256) * cos1 - width_adj + 256
        right2 = (my_vertices['right'][zon]-256)* cos1 +width_adj + 256
        
        if left2 > right2:
            left2, right2 = right2, left2

        height_scale = (660-top_border)/660

        upper2 = top_border + my_vertices['upper'][zon] * height_scale
        lower2 = top_border + my_vertices['lower'][zon] * height_scale

        l2= int(left2+.5)
        u2 = int(upper2+.5)
        w2 = int(right2-left2+.5)
        h2 = int(lower2-upper2+.5)
        zcl = [u2,l2,h2,w2]  #crop list.
        zsl= np.array([[left2,upper2],[right2,upper2],[right2,lower2],[left2,lower2]], np.int32) 
    return zsl


def get_zone(infile, zone,view, filetype):
    #global zsl,an_img,zcl #diag
    an_img = get_single_image(infile, view)
    img_rescaled = convert_to_grayscale(an_img)
    img_high_contrast = spread_spectrum(img_rescaled)
    zsl= zoneslicelist1(zone,view,filetype) #zcl retunred as global 
    masked_img = roi(img_high_contrast, zsl)
    cropped_img = crop(masked_img, zcl)  
    #resize the image?
    return  normalize(cropped_img)


#twist a slanted body part (upper arsms and forearms to a vertical)
def twister(image,hFactor,vFactor,param):
    
    #print("twister image shape", image.shape,"HF", hFactor,"VF", vFactor)
    #global image1,image2,image3, u,v,lr,tb,rCols,rRows,row #diag
    mq=108
   
    if vFactor==1:
        #rows remain same while columns expand
        rCols =  int(nCols*hFactor+.5)
        lr=int(center_row* hFactor +.5) #for 30 try 2
        image1 = cv2.resize(image, (rCols,nRows))
        image2=np.zeros((nRows,rCols+2*lr))

        for row  in range(nRows):
            u= int(lr+(row-center_row)*hFactor+.5)
            v=u+rCols #so when u is 0 v is 1024
            image2[row,u:v]= image1[row,:]

            image2[row, lr+center_column-mq:u] = image1[row,0]
            image2[row, v+1:lr+center_column+mq] = image1[row,-1]

        image3 = image2[:,lr+center_column-mq: lr+center_column+mq] #trial

    elif hFactor==1:
        #cols remain same while rows expand
        rRows =  int(nRows*vFactor+.5)
        tb= int(center_column * vFactor+.5)
        image1 = cv2.resize(image, (nCols,rRows))
        image2=np.zeros((rRows+2*tb,nCols))

        for col  in range(nCols):
            u= int(tb+(col-center_column)*vFactor+.5)
            v=u+rRows #check
            image2[u:v,col] = image1[:,col]
            #padding
            image2[tb+center_row-mq:u, col] = image2[0, col]
            image2[v+1:tb+center_column+mq, col] = image2[-1, col]
            
        image3 = image2[tb+center_column-mq: tb+center_column+mq,:]
        #we preserve 210 columns (before side cropping)
    else:
        image3=image
        print("bad factor inputs",hFactor,vFactor)
    if param=="R":  #I was planning to do the oppiste, but all the arms were coming out horiz
        image3=image3.transpose()
    

    return image3



def colvar(image,angletwist):
    global nCols,center_row,center_column,nRows,im3
    nRows,nCols=image.shape[0],image.shape[1] #make sure doing same
    
    center_row=int(nRows/2+1)
    center_column=int(nCols/2+1)

    #needs to handle dispacelment to left or right:
            
    if angletwist <=45:
        hFactor= (90/angletwist -1)
        im3= twister(image,hFactor,1,"R")

    else:
        vFactor= angletwist/(90-angletwist)
        im3= twister(image,1,vFactor,"T") #sort of a 90deg rotation so straight up

    #to calculate the optimal angle rotation, we ignore the top 20 and bottom rows
    #     
    try:
        im_gist= im3[20:-20,:]
    except:
        im_gist=im3
    computn = np.sum(np.var(im_gist, axis=1)) #verify I have right axis
    #plOT_image(im3)
    #print("computation=",computn)
    return im3, computn


def cropNsave(image,view,zone,eachPsgr): 
    try:
        image2 = side_crop(image,zone)
        image2 = np.where(image2>.19,image2,0) #new in V96
        
        #print("side cropped")
        #plOT_image(image2)
        np.savetxt(PPZV + eachPsgr +  "_z" + str(zone+1)+  "_v"+str(view)+"_.csv", image2, delimiter="|")

    except:
        print("skipping",eachPsgr, view,zone)


def armistice(Psgr,zone):
    #handles forearms and upperarfms
    #choose best angle based on vw0 (to minimize the column variancce.
    #apply that angle to the other views)
    #as a precursor, always try and orient the arm like a backslash (up and to the left)
    
    #zone in armistice, cropnsave etc runs 0 thru 16 not 1 thru 17
    #global bestAngle,at, vw #diag
    
    MYFILE = getfile(Psgr,ftyp)     
#    angletwists = [24,31,38, 45,53,61, 68] #intentional assymetry
#    #a=0
#    save,imX = [],[]
    image = get_zone(MYFILE, zone,0,ftyp) #verify zone counts from zero    
    if zone in [1,2]:
        image = np.fliplr(image)
    
#    for j,Atwist in enumerate(angletwists):
#       imX0, save0 = colvar(image, Atwist)
#       save.append(save0)
#       imX.append(imX0)
#    
#    bestAngle=save.index(min(save)) #choose the angle which minimizes column variance
#    at = angletwists[bestAngle] 
    cropNsave( image ,0,zone,Psgr ) #cropNsave( imX[bestAngle] ,0,zone,Psgr )
     
    for vw in [8,16,24,32,40,48,56]:
        #vwx="vw"+str(view).zfill(2)    
        
        if (zone < 2 and  vw==16) or (zone >=2 and vw==48):
            #cant see from opposite side
            pass
        elif fdict[vw]==0:
            #we have a straight on view from the side. do not rotate
            imOriginal= get_zone(MYFILE, zone,vw,ftyp) 
            cropNsave( imOriginal,vw,zone,Psgr)
            
        else:

            imOriginal = get_zone(MYFILE, zone,vw,ftyp)

            
            #flip lr oonce for lforearm and r upperarm 
            #and flip again if vw is 24 32 40 (in lieu of making fdict negative)
            if zone in [1,2] and vw in [56,8]:
                imOriginal = np.fliplr(imOriginal)
            elif zone in [0,3] and vw in [24,32,40]:
                imOriginal = np.fliplr(imOriginal)

            #twist the view using same angle(at) as in view0
          
#            if at <=45:
#                hFactor= (90/at -1) * fdict[vw]
#                #print (hFactor,"hf",zone,"zone")
#                imRotated = twister (imOriginal,hFactor,1,"R")
#                cropNsave(imRotated,     vw,zone,Psgr)
#            else:
#                vFactor= 2 * fdict[vw]
#                imRotated = twister (imOriginal, 1,vFactor,"R")
#                cropNsave( imRotated,vw,zone,Psgr)
            cropNsave(imOriginal,     vw,zone,Psgr)

#currently breaks on first detection ie as soon as qscore exceeds 0
def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)

def check_shaPe(x,y,xu,yu,param):
    #test to see whther the core is encircles
    #looking ideally for a light or dark core encircled by natural
    #or (lower mult)  for  a natural core encircled by light or dark
    #use binaryfillholes
    #Score 3*mult if the test is satisifed without having to dilate,2*mult if 1 dilation is needed
    #print("CS1",time.time() )
    rad=30 #tunable
    x1 = max(0,x-rad)
    x2 = min(x+2*xu+rad,d1-1)   
    y1 = max(y-rad,0)   
    y2 = min(y+2*yu+rad, d2-1) 
    #if y2>=130:         print(x1,x2,y1,y2,"diag",d1,d2)

    struct3=np.ones((x2-x1+1, y2-y1+1)  )
    #if y2>=130:         print("aa")
 
    #if y2>=130:         print("bb")
    if param=="Light":
        wide_bin0 = bin_dark[x1:x2+1,y1:y2+1] 
        #mult=.9
    elif param=="Natural":
        wide_bin0 = bin_xtrm[x1:x2+1,y1:y2+1]
        #mult=.7 #tune this parameter later
    elif param=="Dark":
        wide_bin0 = 1- bin_dark[x1:x2+1,y1:y2+1] #1 means <.25
        #mult=1.1
    #if y2>=130:         print("cc")
    #wide_bin1 = ndi.binary_dilation(wide_bin0, structure=st3).astype(int)
    wide_bin2 = ndi.binary_dilation(wide_bin0, structure=st3,iterations=2).astype(int)
    
    if   ndi.binary_fill_holes(wide_bin2,structure=struct3).astype(int)[x-x1,y-y1] == 0:
        return 0
    return 1

def scoring (z1,z2,lattice,localavg,localmax,localmin):
    #break as soon as the pzv registers any hit
    for b1 in range(0,d1-2*z1,lattice):
#        if b1 % 10 == 0:
#            print("checking row ",b1)
        for b2 in range(0,d2-2*z2,lattice):

            #b1 and b2 are the coords of upper left corner 
            if localavg[b1,b2]<.15 and localmax[b1,b2]<.2: 
                qscore = check_shaPe(b1,b2,z1,z2,"Light")
                if qscore:
                    return qscore
            elif localmin[b1,b2]>.4:  # the inner rectangel is dark
                qscore = check_shaPe(b1,b2,z1,z2,"Dark")
                if qscore:
                    return qscore
            elif localmin[b1,b2]>.25 and localmax[b1,b2]<.4: # the inner core is natural
                qscore = check_shaPe(b1,b2,z1,z2,"Natural")
                if qscore:
                    return qscore
    return qscore

def find_shaPe(zone_img,z1,z2,lattice):
    #print("FR1",time.time() )

    shape1  = (2*z1+1, 2*z2+1) #height,width
    
    convoluted =  rolling_window(zone_img, shape1)  
    # if the orig image is 70x200 pixels it returns 71-hgt x 201-wdt x hgt x wdt.
    #think of the first 2 dims  as an array of all possible windows
    localavg =np.mean(convoluted, axis=(-1,-2))  #could weight this unfirom filter
    localmax =np.amax(convoluted, axis=(-1,-2))  
    localmin =np.amin(convoluted, axis=(-1,-2))  
    # optimize memory
    
    #alt localmax = ndi.filters.maximum_filter(zone_img, footprint=shape, mode='nearest' ) 
    #if use that check if that filter is centered properly ¶. origin paramter. also do we need the output param
    #https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.filters.maximum_filter.html

    #print("FR2",time.time() )   
    return scoring (z1,z2,lattice,localavg,localmax,localmin)


def Vol2(bA,iter,stP,stQ):
        bB = ndi.binary_dilation(bA, structure=stP,iterations=iter).astype(int)
        bC = ndi.binary_erosion(bB, structure=stP,iterations=iter).astype(int)
        bD = ndi.binary_fill_holes(bC,structure=stQ)
        return bB,bC,bD

def VolumeLab(img0,azon,aview):
    #global hist1,savx,wt,ct,median, bA,bB,bC,bD,bE,bF,bG,bH,rr,str,out,v1,v2,pzidx  #diag
    #global convoluted7
    #returns 1 for skipped records and 0 for useful

    #repeat this for (1) differnt fill holes structures  (2) obverse     #those well below median
    img = np.nan_to_num(img0)
    stP = np.array([[False,  True, False], [True, True, True], [False, True, False]], dtype=bool)

    binH = [.05*x for x in range(15)]
    wt=np.array([0,0,0,0, .225,.275,.325,.375,.425,.475,.525,.575,.625,.675])
    ct=np.array([0]*4+[1]*10)
    hist1=np.histogram(img, bins=binH)[0]
    myct = np.dot(hist1,ct)


    if myct>0:
        dimsum = img0.shape[1]
        if dimsum<7:
            DhumpFile.write("|empty or empty") 

            return 1


        convoluted5 =  rolling_window(img, (5,5) )  
        #if the orig image is 70x200 pixels it returns 71-hgt x 201-wdt x hgt x wdt.
        #think of the first 2 dims  as an array of all possible windows
        
        median=np.dot(hist1,wt)/myct #excl values below .2

        ab1=img*0
        ab1[2:-2, 2:-2] = np.std(convoluted5, axis=(-1,-2))
        localsd5 =np.where(img>.23,ab1,0)
        lavg5= img 
        lavg5[2:-2,2:-2] = np.mean(convoluted5, axis=(-1,-2))

        if dimsum>6:
            convoluted7 =  rolling_window(img, (7,7) ) 
            localvar =np.amax(convoluted7, axis=(-1,-2)) - np.amin(convoluted7, axis=(-1,-2))
            avg=np.average(localvar)
            ab=img*0
            ab[3:-3, 3:-3] = np.std(convoluted7, axis=(-1,-2))
            localsd =np.where(img>.23,ab,0)

            lavg2= img *1
            lavg2[3:-3,3:-3] = np.mean(convoluted7, axis=(-1,-2))
        else:
            localvar =ab1
            localsd=ab1
            lavg2=ab1
            avg=ab1
      
            
        bA  = (median+ .15<img).astype(int) #binary transform
        bB,bC,bD = Vol2(bA,2,stP,st5)
        
        bE  = (((median- .12)>img) &  ((median-.27)<img )).astype(int) #binary transform
        bF,bG,bH = Vol2(bE,2,stP,st5)

        rr1= [np.sum(x)  for x in [bA,bB,bC,bD,bE,bF,bG,bH]]

        bA  = (median+ .2<img).astype(int) #binary transform
        bB,bC,bD = Vol2(bA,1,stP,st5)
        
        bE=(localvar>avg).astype(int)
        bF=ndi.binary_fill_holes(bE,structure=st5)
        bG=ndi.binary_fill_holes(bE,structure=st3)
        bH=localvar>.4
        rr2= [np.sum(x)  for x in [bA,bB,bC,bD,bE,bF,bG,bH]]

        #3 iterations   
        bA  = (median+ .15<img).astype(int)
        bB,bC,bD = Vol2(bA,3,stP,st5)
        
        bE = img-lavg2 >.09
        bF,bG,bH = Vol2(bE,3,stP,st5)
      
        rr3= [np.sum(x)  for x in [bA,bB,bC,bD,bE,bF,bG,bH]]


        bA  = localsd > .07
        bB = localsd > .14
        bC = localsd > .21
        bD = localsd > .28
        bE=ndi.binary_fill_holes(bB,structure=st7)
     
        bF=(localvar>avg).astype(int)
        bG=ndi.binary_fill_holes(bF,structure=st7)
        bH=ndi.binary_fill_holes(bF,structure=st9)
        
        rr4= [np.sum(x)  for x in [bA,bB,bC,bD,bE,bF,bG,bH]]

        bA  = ((median+ .12)<img * (localsd5<.07))*1
        bB  = ((median+ .12)<img * (localsd5<.035))*1
        bC  = ((median+ .24)<img * (localsd5<.07)) *1
        bD  = ((median+ .24)<img * (localsd5<.035))*1
        
        bE  = ((median- .12)>img * ((median-.24) <img)  * (localsd5<.07)) *1 
        bF  = ((median- .12)>img * ((median-.24) <img)  * (localsd5<.035)) *1 
        bG  = ((median>img)  * ((median-.12) <img)  * (localsd5<.07)) *1 
        bH  = ((median>img)  * ((median-.12) <img)  * (localsd5<.035)) *1 
        
        rr5= [np.sum(x)  for x in [bA,bB,bC,bD,bE,bF,bG,bH]]

        if dimsum>8: #occasionally the window is too narrow
            convoluted9 =  rolling_window(img, (9,9) )  
            ab=img*0
            ab[4:-4, 4:-4] = np.std(convoluted9, axis=(-1,-2))
            localsd9 =np.where(img>.23,ab,0)
            lavg9= img 
            lavg9[4:-4,4:-4] = np.mean(convoluted9, axis=(-1,-2))
    
            bA  = ((median+ .12)<img  * (localsd9<.07)) *1 
            bB  = ((median+ .12)<img  * (localsd9<.035)) *1 
            bC  = ((median+ .24)<img  * (localsd9<.07)) *1 
            bD  = ((median+ .24)<img  * (localsd9<.035)) *1 
            
            bE  = ((median- .12)>img  * ((median-.24) <img)  * (localsd9<.07)) *1 
            bF  = ((median- .12)>img  * ((median-.24) <img)  * (localsd9<.035)) *1 
            bG  = ((median>img)  * ((median-.12) <img)  * (localsd9<.07)) *1 
            bH  = ((median>img)  * ((median-.12) <img)  * (localsd9<.035)) *1 
    
            rr6= [np.sum(x)  for x in [bA,bB,bC,bD,bE,bF,bG,bH]]
        else: 
            rr6  = [0]*8

          
        rr=rr1+rr2 +rr3+rr4+rr5+rr6
        v1 = int(aview*vars_per_zonview/8. +3.5)  #first 3 colums reseved for index, zone, view 
        v2=v1+48

        MetrikTable[pz2idx,v1:v2] =np.array(rr)

        out = "|".join([str(x) for x in rr])
        DhumpFile.write("|%s" % out) 
        return 0 #use
        
    else:
        DhumpFile.write("|empty") 
        return 1 #skip
        



def ArmWaveHelper(ii,param1): #maksked array
    #column refers to the 2nd dimension
    global metric
    global cAbove, cBelow,metric,em,ev,ev2,rx #diag

    if param1=="M":
        rx=imgM
    else:
        rx=imgT
    em = np.ma.mean(rx,axis=0)#returns 1d array of column means
    ev = np.ma.std(rx,axis=0) #verify
    ev2=np.ma.power(ev,2)
    metric[ii] = np.ma.sum(ev2)
    metric[ii+1] = np.ma.sum(np.abs(ev))
    metric[ii+2] = np.ma.std(em)
    metric[ii+3] = np.ma.std(ev)
    
    #looking for consecutive vertical blocks which exceed std dev
    #the longer the block the more weight
    #we could try differenet structuring elements
    #we could sum cAbove and cBelow differently

    cAbove = rx>(em+ev)   
    #check how it works with make. 
    cBelow = rx<(em-ev) 
    metric[ii+4] = np.ma.sum(cAbove) 
    metric[ii+5] = np.ma.sum(cBelow)

    for j in [1,2,3]:
        cAbove  = ndi.binary_erosion(cAbove,structure=stC).astype(int)
        cBelow  = ndi.binary_erosion(cBelow,structure=stC).astype(int)
        metric[ii+ 4+ 2*j] = np.ma.sum(cAbove)
        metric[ii+ 4+ 2*j+1] = np.ma.sum(cBelow)



def ArmWave(img0,azon,aview): # array
    
    global metric, imgM, imgT
    global imgM,pointer,imx,vacuum #diag
    
    vacuum=np.sum(img0)
    #print("vacuum",vacuum)
    
    if vacuum<200:
        DhumpFile.write("|empty") 
        return 1
    else:
        imx=img0 #diag
        metric = np.zeros(48)
    
        for pointer in [0, 12]:
            lowthresh= pointer/120 +.22 #clumsy dictionary for .22 to .32
            mask1=img0<lowthresh
            imgM = np.ma.masked_array(imx, mask=mask1,copy=True)
            imgT = imgM.transpose()
    
            ArmWaveHelper(pointer,"M")
            ArmWaveHelper(pointer+24,"T") #verify                
            v1 = int(aview*vars_per_zonview/8. +3.5)  #first 3 colums reseved for index, zone, view 
        MetrikTable[pz2idx,v1:v1+48] = metric
        DhumpFile.write("|%s" %  "|".join([str(x) for x in metric])  ) 
        return 0
           

def pz_indexer(pz,parame):
    pzidx = pzList.index(pz)
    lbl = probList[pzidx]
    if parame !="H":
        try:
            pz2idx = pz2List.index(pz)
        except:
            pz2idx = -1
    else:
        try:
            pz2idx = pz2ListH.index(pz)
        except:
            pz2idx = -1        


    return pzidx,pz2idx,lbl



def post_vol(rows):

    [q0,q1,q2,q3,q4,q5,q6,q7,q8,q9] = [3+48*x for x in range(10)]

    #add columns for max of all views
    MetrikTable[:,q8:q9]= np.maximum.reduce  ([MetrikTable[:,q0:q1], MetrikTable[:,51:q2],
            MetrikTable[:,q2:q3],MetrikTable[:,q3:q4],MetrikTable[:,q4:q5],
            MetrikTable[:,q5:q6],MetrikTable[:,q6:q7],MetrikTable[:,q7:q8]]) 
    zon1= MetrikTable[:,2].astype(int)
     
    row2=min(rows,int(file_lim/5.5)) #theor 6.23
     
    #replace obscured views with average of good views
    for j in range(row2):
        zonj=zon1[j]
        
        #print ("zone",j,zonj)

        if zonj in [1,3,6,8,11,13,15]: #left side
            MetrikTable[j,q2:q3] = (MetrikTable[j,q0:q1] +MetrikTable[j,q1:q2]++MetrikTable[j,q3:q4] +
            MetrikTable[j,q4:q5]+MetrikTable[j,q5:q6]+MetrikTable[j,q6:q7]+MetrikTable[j,q7:q8])/7

        elif zonj in [2,4,7,10,12,14,16]: #right side
            MetrikTable[j,q6:q7] = (MetrikTable[j,q0:q1] +MetrikTable[j,q1:q2]+MetrikTable[j,q2:q3]+MetrikTable[j,q3:q4]+
            MetrikTable[j,q4:q5]+MetrikTable[j,q5:q6]+MetrikTable[j,q7:q8])/7

        elif zonj ==5:#front chest
            for vw in [8,16,24,32,40,48,56]:
                QA = q0+6*vw  #5 = 48 varaiables dovoded by 8 scalar in views
                QB = q1+6*vw
                MetrikTable[j, QA:QB] = MetrikTable[j,q0:q1]
                #print("avg=",avg)

        elif zonj ==17: #bacl

            for vw in [0,8,16,24,40,48,56]:
                QA = q0+6*vw
                QB = q1+6*vw
                MetrikTable[j,QA:QB] =MetrikTable[j,q4:q5]     
                     
        elif zonj==9: #groin
            avgG = (MetrikTable[j,q0:q1] +MetrikTable[j,q1:q2]+MetrikTable[j,q3:q4]+
                    MetrikTable[j,q4:q5]+MetrikTable[j,q5:q6]+MetrikTable[j,q7:q8])/6
            for vw in [16,48]:
                QA = q0+6*vw
                QB = q1+6*vw
                MetrikTable[j,QA:QB] =  avgG 
        else:
            print(j,"bad zone",zonj)

    

def algoQ(zone_img,azon,l1,l2,l3): 
    #azon for Zone8 is 8 not 7
    #zone17 picks up z5 v32
    
    score=np.zeros(3) #elim the 4th
    #score[3]=  check_hatch(zone_img)    
    global bin_dark,bin_xtrm,d1,d2,hist1 
    (d1,d2) = zone_img.shape
    
    #VolumeLab(zone_img)
    
    #if azon > 10 or azon in [ 3, 1, 7, 6]: 
    
    # upperarms,  lowerchest, back, legs (knee ,below, and calf, )
    #dont run the most intensive search on the 
    
    bin_dark =(zone_img>.25).astype(int)
    bin_xtrm  = (.15>zone_img).astype(int)+(.45<zone_img).astype(int) #changed from myimg. verify

    #lattice speed vs efficacy
    #skip any run which has already tested positive on previous views
    if l1:
        score[0]=find_shaPe(zone_img,3,3,4) #7x7
    if l2:
        score[1]=find_shaPe(zone_img,5,1,3) #11x3
    if l3:
        score[2]=find_shaPe(zone_img,1,5,3) #3x11

    return score


def RunMetrix(p2LST,filnam,dump1,param,XYZ):

    
    global MetrikTable, file_lim, vars_per_zonview,DhumpFile,Ymask

    global probList #verify if needed
    global pz2idx,ktr,pzv,pzidx,label,PZVlist,pz #diag
    
    file_lim = 200000 #500 in trail mode
    
    dimx=len(p2LST) 
    #passList = pzInvty["Passenger"].tolist() 
    zonList = pzInvty["Zone"] .tolist()
    probList = pzInvty["Probability"].tolist()  #read as integer?
    vars_per_zonview = 48
    MetrikTable=np.zeros((dimx,3+9*vars_per_zonview))  
    Ymask=np.ones(dimx)  
    # ID, label, zone,  + 8 views x 16 vars + max column
    with open(dump1,"w") as DhumpFile: #vary for hidden run?
    
        PZVfiles =   pd.read_csv(PATH1 + r"/file_inventory/pzv_invty_20171128.csv",delimiter = "|")
        PZVlist = PZVfiles["pzv_file"].tolist()

    
        #review each passenger zone views 
        for ktr,pzv in enumerate(PZVlist[0:file_lim]):
            
            pz = pzv.strip().rsplit("_",2)[0]
            pzidx,pz2idx,label = pz_indexer(pz,param)
            
            #print("label=", label)
    
            if pz2idx >=0:
           
                DhumpFile.write("\n%s" % pzv) 
                record=PZVfiles.iloc[ktr]
                view = int(record['view']) #the sourcefile has integers such as 0,8,16
                 
                #zoneQ corresponds to MYzone=Zone1; z1 in filename, PZVfiles and record; and 1 in my_vertices
                #there may be contexts where "zone" is one less, I have tried to eradicate those
                
                #special handling of back   
                if record['zone']==5 and view==32:
                    zoneQ = 17 
                    #print("pp",pz,record,pzidx)
                    pz17 = pz.replace("z5","z17")
                    #print("pz17",pz17)
                    pzidx,pz2idx,label = pz_indexer(pz17,param)
                    #print("zoneQ",zoneQ)
                else:
                    zoneQ = record['zone']
                #print("==", pz2idx,pzidx,label,zonList[pzidx]) 
                

                if param=="H":
                    label=-1
                else:
                    label=int(label)

                #MetrikTable col
                MetrikTable[pz2idx,:3] = np.array([ pzidx, label,  zonList[pzidx]])
    
                #passr= record['passgr']
        
                #progress printout,
                if ktr%500 == 0: 
                    print(ktr,time.time(),end="; ")
                
                #there is probably a more pythonic way 
                with open( PPZV  +  pzv + ".csv",'r') as inp_f:
        
                    #try:
                        data_iter = csv.reader(inp_f,delimiter = "|")
                        data = [x for x in data_iter]
                        myimg = np.asarray(data).astype(np.float)
                        if XYZ=="48":
                            shouldmask=VolumeLab(myimg,zoneQ,view) 
                            if ~shouldmask: #if any good image exists, dont mask it
                                Ymask[pz2idx]=0 #0 means in use
                        elif XYZ=="59" and zoneQ in [1,2,3,4,5,9,17]:
                            shouldmask=ArmWave(myimg,zoneQ,view) 
                            if ~shouldmask: #if any good image exists, dont mask it
                                Ymask[pz2idx]=0 #0 means in use


    print("\npost_processing...")
    post_vol(dimx)
    np.savetxt(filnam, MetrikTable.astype(int), fmt='%i', delimiter="|")



def AfterMetrix(MT, xyz):
    import pickle
    global logr
    global y,X,Ymask, xdim,Xmask,yM,XM, yMG, XMG,useful, pz3List ,pred1 #diag
    #global pkl_filename, pz2List #diag
    
    y,X = MT [:,1], MT [:,2:] #TT
    from sklearn.model_selection import train_test_split

    useful= y.shape[0] - int(.5+sum(Ymask))# counts the number of zeros TT

    xdim= X.shape[1]
    Xmask= np.outer(Ymask, np.ones(xdim) )
    
    #grand masked array
    yMG = np.ma.masked_array(y,mask=Ymask )
    XMG = np.ma.masked_array(X,mask=Xmask) 
    #compressed masked array
    yM=np.ma.compressed(yMG).ravel()
    XM=np.ma.compressed(XMG).reshape(useful,xdim)
    
    
    # Split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(XM, yM, test_size = .2,
                                        random_state=42, stratify=yM)
 
    #http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    from sklearn.linear_model import LogisticRegression as LogR  
    logr = LogR (penalty="l2",tol=.0001, class_weight=None,
          random_state=None, solver="liblinear",verbose=0,max_iter=200)
    #models like sag, higher tolerance, didnt affect score. l1 jirt
    
    # fit(X, y[, sample_weight])	Fit the model according to the given training data.
    logr.fit(X_train, y_train)

    #y_pred=knn.predict(X_test)
    pkl_filename = PATH1 + "/Pickled/pkl_"+xyz+"metric_model.pkl"

    with open(pkl_filename, 'wb') as outFile:  
        pickle.dump(logr, outFile)
    
    pred1= logr.predict_proba(XM)[:,1]	#Probability estimates.
    pz3List = np.array([pzInvty[pzInvty["PZ"]==x]["PZidq"].values[0] for x in pz2List]) 
    pz3Lm = np.ma.masked_array(pz3List,mask=Ymask ).compressed()
    new = np.zeros((len(pred1),3)).astype(str) 
    new[:,0] = np.array(pz3Lm)
    new[:,1] = np.char.mod('%d', yM) #actual prob
    new[:,2] = np.char.mod('%1.8f', pred1)
    
    np.savetxt("predict"+xyz+".csv", new, delimiter="|",fmt = "%s" ,header="pzvID|label|prediction") #saves predictions which are needed in 16

    n, bins, patches = plt.hist(pred1, 100, facecolor='blue', alpha=0.5) #logscale
    #y = mlab.normpdf(bins, mu, sigma)

    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.title(r'Distribution')
     
    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()
    
   
    print("mean accuracy: ",logr.score(X_test, y_test))

    from sklearn.metrics import r2_score 
    print("r_squared: ", r2_score(yM, pred1))
    #additional statistics on the quality of the fit
    #slope, intercept, r_value, p_value, std_err = stats.linregress(pred1, y)


def getfile(each,ftyp):
    if ftyp==".aps":
        MYFILE = PATH1 + r"/TSAimages/" + each + ftyp
    elif ftyp == ".a3daps":
        MYFILE = EnvX + r"/_A3DAPS/" + each + ftyp

    else:
        MYFILE = EnvX+ r"/_A3D/" + each + ftyp
    return MYFILE

def side_crop(img,zone):
    #global light_cols #diag
    widthm1 =img.shape[1] -1
    
    # crop dark columns from left or right of image
    #light columns are defined based on the max value and to a lesser extent the average value
    
    lcM = np.amax(img,0)  #col max 
    lcA = np.mean(img,0)  #col avg
    if zone in [1,3]: #1 and 3 correspond to zones 2 and 4
        light_cols = (lcM>.22)|( (lcM>.2) & (lcA>.14) )   #light columns
    else:
        light_cols = (lcM>.2)|( (lcM>.18) & (lcA>.12) )   #light columns
    left_border =  np.argmax(light_cols)  #find first light column
    right_border = widthm1- np.argmax(light_cols[::-1]) #find last light column (by reversing the string) 
    l1= max(left_border-2, 0)
    r1= min(right_border+2,widthm1)
    return img[:, l1:r1  ]        

def cpx(file,dest):
    try:
        cp(pzvFile, dest)
    except:
        print("error copying",file,"t",dest)

def pp_saver1(qFile,label,pz_id,pssgr):
    global savaBatch,savHidden

    try:
        image = np.genfromtxt(qFile,delimiter='|')
        if label !=[2,-1]:
            savaBatch.append([pz_id,label,image ]) 
            #save even worthless images, it might impede the training but we want them in the prediciton file
            #fix consider excluding nonpristine views                            
            #NB although we save the pzvid we dont care whether it is 5 or 17
        else:
            savHidden.append([pz_id,image ]) #no need to flio  
    except:
        print("error_missing",qFile)

#selects relevant images for a region (generally consisting of 2 symmetric zones) 
#and stores them in mini batches
def MiniBatch(region):
    #global savaBatch,savHidden,pzSubset,pzMinibatch #diag?

    # dont exclude images even with poor quality if img.shape[0]>6 and img.shape[1]>6:
    rz_dict = {"hip":[8,10], "knee":[11,12],"foot":[15,16], "torso":[6,7],"forearm":[2,4],
               "upperarm":[1,3], "calf":[13,14],   "groin":[9,-1],"upperchest":[5,-1],"myachingback":[17,-1],"uchback":[5,17]   }
    #-1 signifies there is just one zone

    label_dict = {"1": [0,1], "0": [1,0],"Hidden":[2,-1]}

    zONe1,zONe2 = rz_dict[region][0], rz_dict[region][1]
    
    #read a list of passenger zones and select as pzSubset hips correspoding to the two zones
    pzInvty = pd.read_csv("pz_invty4.csv",delimiter="|") 
    pzSubset =   pzInvty[  (pzInvty["Zone"] ==zONe1)|(pzInvty["Zone"] ==zONe2) ]
    
    tot= pzSubset.shape[0]  
    savHidden=[]
    batches = int(tot/BATCH_SIZE+.99) 

    for batch_num  in range(batches): #batches number from zero
 
        savaBatch = []
        
        #select a  batch of records as pzMinibatch
        u1= BATCH_SIZE * batch_num 
        u2 = min(BATCH_SIZE * (batch_num+1),tot)
        pzMinibatch = pzSubset[u1:u2] 
       
        for index, row in pzMinibatch.iterrows():
            pssgr=row["Passenger"] #verify needed
            zon_num= row["Zone"] #1 to 17
            pzx = row["PZ"]
            PZidq = row["PZidq"] #PPPPzz, suitable for numpy #verify needed
            label=label_dict[row["Probability"]]

            if zon_num ==5:
                    pzvFile=  PPZV +pzx +"_v0_" + ".csv"                    
                    pp_saver1(pzvFile,label,PZidq,pssgr)

            elif zon_num ==17:
                    pzvFile=  PPZV+ pzx.replace("_z17","_z5")   +"_v32_.csv"       
                    pp_saver1(pzvFile,label,PZidq,pssgr) 


            else:
                
                for view in range(0,64,8):  #for a3daps
                    vwx="vw"+str(view).zfill(2)
                    if (my_vertices.loc[zon_num-1][vwx] ==1): 
                      try:
                        pzvFile=  PPZV +  row["PZ"] +"_v" + str(view) + "_.csv"
                        img = np.genfromtxt(pzvFile,delimiter='|')

                                                             
                        if zon_num==zONe2 and zon_num>4: #arms are already fliped for symmetry
                            img = np.fliplr(img)

                        if label !=[2,-1]:
                            savaBatch.append([PZidq,label,img])
                            #we commingle zone1 and zone2 for training 
                        else:    
                            savHidden.append([PZidq,img])

                      except:

                        print("error_missing",pzvFile)

        #save MiniBatch

        outFile=    PPDF + '/' + region +'-b{}.npy'.format(batch_num)

        savaBB=np.array(savaBatch, dtype='object')
        np.save(outFile, savaBB)

        print(' -> writing: '+ outFile+ " length="+ str(len(savaBatch)))

    #after all minibatches have been compiled, save HIDDEN PROB records
    savHH=np.array(savHidden, dtype='object')
    np.save(PPDF + '/Hidden_' + region ,savHH)
    


def get_train_test_file_list():
   
    global FILE_LIST, TRAIN_SET_FILE_LIST, TEST_SET_FILE_LIST



    



    if os.listdir(PPDH) == []:
        print ('No pre-processed data available.  Skipping ...')
        return
    
    FILE_LIST = [f for f in os.listdir(PPDH) if re.search(rgn_search, f)]
        
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


    
    minibatched_scans = np.load(os.path.join(PPDH, filename)) #Load a batch of pre-procd tz scans
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
        
        # run through every batch in the training set
        for ktr,each_batch in enumerate(TRAIN_SET_FILE_LIST):
         
            Batch_pzid, Batch_lb, Batch_im = input_pipeline(each_batch,1)
            Batch_im = Batch_im.reshape(-1, Dm1, Dm2, 1)
    
            # run the fit operation
            #n-epoch=1
            trainX, trainY =  {'imAges': Batch_im},  {'labels': Batch_lb}
            testX,testY = {'imAges': val_images} , {'labels': val_labels}
            # maybe n_epoch = orig val of N_TRAIN_STEPS
            model.fit(trainX, trainY , n_epoch=1, validation_set=(testX,testY), 
                      shuffle=True, snapshot_step=None, show_metric=True, 
                      run_id=MODEL_NAME)
    
            model.save(MODEL_PATH+"m_"+rgn+".tflearn")
            
            #save prediciton file on final pass
            if i == N_TRAIN_STEPS-1: # if one pass, modify or remove this check
                trainPred = model.predict(trainX)
                predict_sv(Batch_pzid, Batch_lb, trainPred,rgn,ktr,"TrainPred")
    
        #after the last batch, save validation set 
        testPred = model.predict(testX)
        predict_sv(val_pzids, val_labels, testPred,rgn,ktr,"TestPred")

def trainer(rgn):
    
    #TFLearn treats each "minibatch" as an epoch.   
    global eachfile, AllTrainingPredicts, HiddenPredictions, Hid_pzid
    global val_images, val_labels, model,trainX, trainY,testX,testY , val_pzids
    #global trainPred, testPred  #diag
    #global flag,Batch_pzid,  #diag
    #global Batch_im, Batch_lb,Hid_im , tmp_image_batch #diag


    #print("currently myachingback uses the smaller batch size" )


  
    val_images,    val_labels = [],[]

    get_train_test_file_list()  # get train and test batches
    #defines FILE_LIST, TrainingFiles, TEST_SET_FILE_LIST
    tf.reset_default_graph()

    model = D1N(Dm1, Dm2, Learning_Rate)    # instantiate model        

    
    # read in the validation test set
    for j, test_f_in in enumerate(TEST_SET_FILE_LIST):
        if j == 0:
            val_pzids, val_labels, val_images  = input_pipeline(test_f_in,1)
        else:
            tmp_pzid_batch, tmp_label_batch, tmp_image_batch,  = input_pipeline(test_f_in,1)
            
            val_pzids  = np.concatenate((tmp_pzid_batch,  val_pzids),  axis=0) 
            val_images = val_images.reshape(-1, Dm1, Dm2) #needed for back when multiple validation batches??
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

    Hid_pzid, dummy, Hid_im = input_pipeline(PPDH + '/Hidden_' + rgn+".npy",2) #hatch-lb is zeros
    Hid_im = Hid_im.reshape(-1, Dm1, Dm2, 1)
    HiddenPredictions=model.predict({'imAges': Hid_im})
    HiddenPredictions[:,0]=Hid_pzid #overwrite column zero

    np.savetxt(PREDICT_PA2 + "/compositeH_"+ rgn +"_.csv", HiddenPredictions, delimiter="|")
    print("saved compsite Hidden single image pred file with shape ",HiddenPredictions.shape)              

#network consists of 5 convolutions/maxpools layers plus 2 regression layers at the end.
def D1N(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='imAges')
    
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

    momentum = Momentum(learning_rate=lr, lr_decay=0.99, decay_step=1000,staircase=True)    
    #Momentum (momentum=0.9,  use_locking=False, name='Momentum')
    #momentum='momentum'

    network = regression(network, optimizer=momentum, loss='categorical_crossentropy',  #layer7
                         learning_rate=lr, name='labels')


    model = tflearn.DNN(network, checkpoint_path=MODEL_PATH + MODEL_NAME, 
                        tensorboard_dir=TRAIN_PATH, tensorboard_verbose=3, max_checkpoints=1)

    return model




def BuildRegrTbl(param,XP,zdim):
    #builds regression table using the 3 highest scroing ML views and the 48 metrics
    #for most zones (zdim=9)10 colums (0 to 7 for data, 8 for 48predand 9 for combined predicitons)
    #for Nettlesome zones(zdim=10), 9 is for 59pred and 10 is for combined pred. tot of 11 columns
    global allPZID,prob48,p48 ,pred,itemindex#diag
    col_dict = {"T":4, "H":1} #the training and validation sets have 5 columns, the Hidden have just 2

    allPZID  = [int(x+.5) for x in  sorted(set( XP[:,0])) ]
    Max3Table = np.zeros( (len(allPZID),zdim+1)   ) #new in 82. add column for 48Metric predictions
    Max3Table[:,0] = allPZID
    
    if param=="T":    
        p48 = pd.read_csv(PATH1 + "/predict48.csv",delimiter="|").set_index('pzvID')
        prob48 = p48["prediction"]
    elif param=="H":    
        p48H = pd.read_csv(PATH1 + "/predict48H.csv",delimiter="|").set_index('pzvID')
        prob48 = p48H["prediction"]

    if zdim==10:
        if param=="T":    
            p59 = pd.read_csv(PATH1 + "/predict59.csv",delimiter="|").set_index('pzvID')
            prob59 = p59["prediction"]
        elif param=="H":    
            p59H = pd.read_csv(PATH1 + "/predict59H.csv",delimiter="|").set_index('pzvID')
            prob59 = p59H["prediction"]


    for j,eachID in enumerate(allPZID):
       itemindex = np.where(XP[:,0]==eachID)[0]  #turns list of row indices

       if param=="T": #for training and test sets we pluck the label from column c1
           Max3Table[j,1]  =XP[itemindex[0],2] #retrieves the threat label  from XP and placesit in column 1

       pred = XP[itemindex,   col_dict[param] ] 
       
       if Threat_Rgn in  ['myachingback','upperchest']:
           Max3Table[j,2:5] = pred[np.argsort(-pred)][0] #broadcast first element
       else:
           Max3Table[j,2:5] = pred[np.argsort(-pred)][0:3] 

       try:
           Max3Table[j,8]  = prob48[eachID]
       except:
           print("<BuildRegTbl> error retrieving 48metric pred",j,eachID)
       if zdim==10:
           try:
               Max3Table[j,9]  = prob59[eachID]
           except:
               print("<BuildRegTbl> error retrieving 59metric preds",j,eachID)           
           
    #cross terms
    Max3Table[:,5] =  Max3Table[:,2] * Max3Table[:,3]  
    Max3Table[:,6] =  Max3Table[:,2] * Max3Table[:,4]  
    Max3Table[:,7] =  Max3Table[:,3] * Max3Table[:,4]  
    

    return Max3Table

def fmt2ID(str):

    #0367394485447c1c3485359ba71f52cb_Zone6
    pzv1=str.replace("_Zone","_z")
    return pzInvty.loc[pzv1]["PZidq"]


def viewCombiner(Threat_Rgn)  :
    

    import pickle    
    global z1,hidden_intermediates,Hidden_Pr,training_intermediates_ML
    global training_labels, training_intermediates_expanded,dim #diag
    global slope, intercept, r_value, p_value, std_err
    

    #requires global input
    #AllTrainingPredicts 
    #combines batches. columsn are pzvid, threat label (two columns) single image prob 2 cols. 
    
    #we only care about columns 0 2 and 4
    
    #z1 col1 is labels. 
    # 0 and 1 are allPZID and ?single image prediction
    # 234 567 are A B C  AB AC BC 
    #column 8 is the 48 metrics prediction
    #for regions upperarm, forearm, groin, upperch, and muyachingback
    #.. col 9 is the 59 metric prediction
    #last colis a placeholder for combined predictions
    
    Nettlesome = ["upperarm", "forearm", "groin", "upperchest", "myachingback"] #fix
    print("override of the Nettlesome")
    if Threat_Rgn in Nettlesome:
        dim=9
    else:
        dim=9
    #dim1=dim-1
   
    
    z1 = BuildRegrTbl("T",AllTrainingPredicts,dim) 
    
    training_intermediates_ML = z1[:,2:8] #PZID label maxview1 maxview2 maxview3 1x2 1x3 2x3 prediciton
    training_intermediates_expanB  = z1[:,2:9] 
    training_intermediates_expanded = z1[:,2:dim] 
    training_labels =  z1[:,1]
    
    from sklearn.linear_model import LogisticRegression as LogR  
    logr = LogR (penalty="l2",tol=.0001, class_weight=None,
          random_state=None, solver="liblinear",verbose=0,max_iter=200)
    print("\nfitting logistic regression..")

    logr.fit(training_intermediates_ML, training_labels)
    print("MaxView coefficients (ML only)",logr.coef_,"intercept",logr.intercept_,sep="|")
    max1_Predict = logr.predict_proba(training_intermediates_ML)[:,1]	
    r_value=np.corrcoef(max1_Predict, training_labels)[1,0]
    print ("R\u00b2 = {0:0.3f}\n".format(r_value**2) ) 
    
   
    logr.fit(training_intermediates_expanB, training_labels)
    print("\nincluding 48 metrics....\nMaxView coefficients" ,logr.coef_,
          "intercept", logr.intercept_, sep="|")
    max2_Predict = logr.predict_proba(training_intermediates_expanB)[:,1]	
    r_value=np.corrcoef(max2_Predict, training_labels)[1,0]
    print ("R\u00b2 = {0:0.3f}\n".format(r_value**2) ) 


    if dim==10:      
        logr.fit(training_intermediates_expanded, training_labels)
        print("\nincluding 48 and 59 metrics....\nMaxView coefficients" ,logr.coef_,
              "intercept", logr.intercept_, sep="|")
        max2_Predict = logr.predict_proba(training_intermediates_expanded)[:,1]	#Probability estimates go in last column
        r_value=np.corrcoef(max2_Predict, training_labels)[1,0]
        print ("R\u00b2 = {0:0.3f}\n".format(r_value**2) ) 


    with open(PREDICT_PA2 + "/logistic_"+ Threat_Rgn +".pkl", 'wb') as outFile:  
        pickle.dump(logr, outFile)


    z1[:,-1] =max2_Predict
    if dim==9:
        np.savetxt(PREDICT_PA2 + "/ViewComb1_Tr_"+ Threat_Rgn +"_.csv", z1, fmt='%d|%d|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f')     
    else: #dim=10
        np.savetxt(PREDICT_PA2 + "/ViewComb1_Tr_"+ Threat_Rgn +"_.csv", z1, fmt='%d|%d|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f|%9.5f')     

  

    hidden_intermediates = BuildRegrTbl("H",HiddenPredictions,dim)[:,0:-1] #omit last colum which is blank
    pzvidE = hidden_intermediates[:,0] #this is pzvid

    raw_prediction =logr.predict_proba(hidden_intermediates [:, 2:dim])
    Hidden_Pr = np.minimum(1,np.maximum(raw_prediction,0)) #dim changed form 9

    
    pzInvty = pd.read_csv(PATH1 + "/pz_invty4.csv",delimiter="|").set_index('PZidq')
    
    hidden= pzInvty[pzInvty["Probability"]=="Hidden"]["PZ"]

    qfile=PREDICT_PA2 + "/ViewComb1_Hd_"+ Threat_Rgn +"_.csv"
    with open(qfile,"w") as outFile:
        outFile.write("\nId,Probability")

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

#if __name__ == "__main__":
#    # Comment the following line to see the default behavior.
#    warnings.simplefilter('error', UserWarning)

  
envt= input("KOI or  MACK:   ").upper()
if envt== "KOI":
    PATH1 = r"C:/Users/Walzer/Documents/DS/Kaggle1/Homeland"     
    EnvX = r"F:"  #C:

elif envt=="MACK":
    PATH1= r"C:/Users/Trapezoid LLC/Desktop/Homeland2"
    EnvX= r"C:"
else:
    print("bad input")
  

INPUT_FOLD2R = EnvX + r"/_A3D"     #INPUT_FOLDER = 'tsa_datasets/stage1/aps'
INPUT_FOLD3R = EnvX + r"/_A3DAPS"     

PPDF =  EnvX + r"/preProcessed"    #contains preprocessed .npy files 
PPDH =  EnvX + r"/preProcessedH"    #contains preprocessed .npy files 

print("overwrite of PPZV")
PPZV = EnvX + r"/_np_zvw2/" # EnvX + r"/_np_zvw2/"

BODY_ZONES =  PATH1 + '/body_zones.png'

COLORMAP = 'pink'
nullZSL = np.array([[0,0],[0,0],[0,0],[0,0]], np.int32) 

global top_border, Bool_U,Bool_D,Bool_L,Bool_R,st3,st4,st5, stC, control

st3 = np.ones((3,3),dtype=bool)  
st4 = np.ones((4,4),dtype=bool)    
st5 = np.ones((5,5),dtype=bool)
st7 = np.ones((7,7),dtype=bool)
st9 = np.ones((9,9),dtype=bool)
stC = np.ones((3,1),dtype=bool)  #verify we dont want the transpose
Bool_U = np.array([[False,  True, False], [False, False, False], [False, False, False]], dtype=bool)
Bool_D = np.array([[False,  False, False], [False, False, False], [False, True, False]], dtype=bool)
Bool_L = np.array([[False,  False, False], [True, False, False], [False, False, False]], dtype=bool)
Bool_R = np.array([[False,  False, False], [False, False, True], [False, False, False]], dtype=bool)


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

vMult = {'.aps': 4, '.a3daps': 1, '.a3d': 1, '.ahi': 1}
rgnlist=["calf","foot","forearm","groin","hip","knee","torso","upperarm",
         "upperchest","myachingback","uchback","allarms"]

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

'''
divide the full image into "sectors", read in the coordinates in front view.
use quasi-trig (coefs in phase_moon) to estimate the coords of that sector when we view from another angle 
Get_zone uses these vertices to isolate image of threat zone in view
sector 5 is used for both threat zone 5 and 17
'''

#uncomment this to show the threat zones 
#body_zones_img = plt.imread(BODY_ZONES)
#fig, ax = plt.subplots(figsize=(15,15))
#ax.imshow(body_zones_img)

#list of passengers with data
df1 = pd.read_csv(PATH1+ r"/passgr_data_invty1.csv",sep="|")
#df2 = df1.loc[df1['small'] != 1]  #handrevised to process only the NEWER files
Pssgr_List1 = df1["passenger"].tolist() 

#df3 = get_hit_rate_stats(THREAT_LABELS) #fix this for Threats_df

top_border  =0
ftyp= ".a3daps"

print(" \n(2)all zones for 1 p \n(3)generate numpy \n(4) save pzv"+
      "\n(5) redon zone17 v \n(7)"+
      "\n(8) generate .pngs\n(12) Metrics"+
      "\n(14) housekeep  \n(15) preProcess \n(16) trainer\n(17) combiner ")
     
control = int(input("control #: "))





elif control==4: 
    #generate pzvs 
    #numpy array for each passegner x 8 views x 17 zones
    #isses with view 16, view 48 , and also view 8/56 x zone 5/7
    print("numpy array for each passegner x 8 views x 17 zones")
    #fdict used for arms rotation. for 24 32 40 we use the abs() here and fliplr below
    fdict={8: 0.71, 16:0,   24:0.71, 32:1.0,   40:0.71, 48:0,  56: 0.71,  0:1  }
    start = 0


    with open("topborderlist.csv","a") as outFile:
        for eachPsgr in Pssgr_List1[start:]: 
            #if eachPsgr in known_missing: #Temp fix
  
            vw00 = get_single_image(getfile(eachPsgr,ftyp), 0) 
            top_border  = top_finder(vw00) #determines passenger_height
            outFile.write('%s|%d' % (eachPsgr,top_border)) 
            print(eachPsgr,top_border)
            

            for zone in range(16): #zone=2 corresponds to ZONE3
                #zone 17 will be  saved as zone 5 vw32

                if zone <=3:
                    vw=armistice(eachPsgr,zone)
                else:
                    for view in range(0,64,8):  #1 to 64 for a3daps
                        MYFILE = getfile(eachPsgr,ftyp) 
                        
                        vwx="vw"+str(view).zfill(2)
                        #zone_index =my_vertices.index[my_vertices["Zone"]==zone+1]
                        if my_vertices.loc[zone][vwx] ==1:
                            vw = get_zone(MYFILE, zone,view,ftyp)
    
    
                            try:
                                vw2 = side_crop(vw,zone)
                                np.savetxt(PPZV + eachPsgr +  "_z" + str(zone+1)+  "_v"+str(view)+"_.csv", vw2, delimiter="|")
            
                                #plOT_image(vw)
                            except:
                                print("skipping",each, view,zone)



elif control==5: #save pzvs for zone17
    print(" numpy array for each passegner x 8 views x 17 zones")
    for each in Pssgr_List1:

        vw00 = get_single_image(getfile(each,ftyp), 0) 
        top_border  = top_finder(vw00) #determines passenger_height
        MYFILE = getfile(each,ftyp) 
        if my_vertices.loc[4]["vw32"] ==1:
            vw = get_zone(MYFILE, 4,view,ftyp)
            try:
                vw2 = side_crop(vw,16) #the 16 refers t zone17
                check = np.sum(vw2)
                if check==0:
                    print (each,vw2)
                else:
                    np.savetxt(PPZV + each +  "_z5_v32_.csv", vw2, delimiter="|")
            except:
                print("skipping",each, 32,4)




       
        
elif control==12:  #run the 48 metrics on all pzvs
    global pzInvty,pzList,pz2List, pz2ListH #verify whether needed

    #global DhumpFile, vars_per_zonview, MetrikTable pzi2dx #not needed
    #global  probList, 

    print("\nRun Metrics\n")
    pzInvty = pd.read_csv("pz_invty4.csv",delimiter="|") #header='infer',  index_col=None,
    pzList = pzInvty["PZ"].tolist()
    pz2List = pzInvty [(pzInvty["Probability"]=="0") | (pzInvty["Probability"]=="1")  ]["PZ"].tolist()  

    print("override of the 59")
    for suffix in ["48"]:#"59"
        RunMetrix(pz2List,"spyd_"+suffix+".csv","dump"+suffix+".csv","R",suffix) #saves intermediate output
        AfterMetrix(MetrikTable, suffix)

        savMetrikTable=MetrikTable
        savYmask=Ymask
        #59 runs ArmWave on a subset of zones
    
        #generate predictio nfor the 100 Hidden
        pz2ListH = pzInvty [ pzInvty["Probability"]=="Hidden"]["PZ"].tolist()  #fix or pzInvty["Probability"]==1
        
        print("\ncompilingmetrics for Hidden passengers")
        RunMetrix(pz2ListH,"Hidden"+suffix+".csv","dumpH.csv","H",suffix)
        X = MetrikTable [:,2:]
        predH= logr.predict_proba(X)[:,1]	#Probability estimates.
        if suffix=="59":
            zoneFilter = ( (MetrikTable[:,2]<=5) | (MetrikTable[:,2]==9) |(MetrikTable[:,2]==17))
            predH=predH * zoneFilter
            
       
        pz3ListH = np.array([pzInvty[pzInvty["PZ"]==x]["PZidq"].values[0] for x in pz2ListH])
        newH=     np.zeros((len(predH),2)).astype(str) #changed from pred1. verify
        newH[:,0] =   np.array(pz3ListH)
        newH[:,1] = np.char.mod('%1.8f', predH) #changed from pred1 verify
        np.savetxt("predict"+suffix+"H.csv", newH, delimiter="|",fmt = "%s",header="pzvID|prediction" )  #changed from new to new H
        #saves predictions, needed in 16
     

        '''#Metrick t
        (0)pzidx (1) label or -1 for Hidden (2)  zone(1to17) followed by 9x48 metris
        '''
     


elif control==15:
    BATCH_SIZE = 312
    
    rgn=''
    print(rgnlist)  
    while (rgn not in rgnlist and rgn not in ["all","custom","hidden"]):
        rgn = input("region?  ").lower()


    if rgn=="all":
        for eachrgn in rgnlist :
            print ("- - - " + eachrgn + " - - - -")
            MiniBatch(eachrgn)
    elif rgn=="custom":
        for eachrgn in "upperarm","forearm":
            print ("- - - " + eachrgn + " - - - -")
            MiniBatch(eachrgn)

    else:
        print ("- - - " + rgn + " - - - -")
        MiniBatch(rgn) 

elif control==16:

    import tensorflow as tf #needed?
    from random import shuffle
    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.optimizers import Momentum
    
    #we call first dim height and second width

    FILE_LIST = []               # A list of the preProcesed .npy files to batch
    TRAIN_SET_FILE_LIST = []     #The list of .npy files to be used for training
    TEST_SET_FILE_LIST = []      #The list of .npy files to be used for testing
    TRAIN_TEST_SPLIT_RATIO = 0.2 #Ratio to split the FILE_LIST between train and test
    TRAIN_PATH = PATH1 + '/DNN/train/'  #Place to store the tensorboard logs
    MODEL_PATH = PATH1+ '/DNN/model/' # Path where model files are stored   

    PREDICT_PA2 = PATH1 + '/DNN/Predict'
    PREDICT_PATH = PREDICT_PA2 + '/P_'

    #Tuneable Parameters
    Learning_Rate = .01  #orig 10^ -3     
    print("override 3 training steps for groin")
    N_TRAIN_STEPS = 10       # The number of train steps (epochs) to run new in 95


    print(rgnlist)
    Threat_Rgn="" 
    while Threat_Rgn not in rgnlist: 
        Threat_Rgn = input("region : ").lower()

    print("=========================================================")
    print ("- - - " + Threat_Rgn + " - - - -     passes=",N_TRAIN_STEPS)
    print("=========================================================")

    print("currently myachingback uses the smaller batch size from a sep directory" )
#    if Threat_Rgn in ["myachingback","groin"]: #new in version95
#        PPDH=PPDH
#    else:
#        PPDH=PPDF

    rgn_search =re.compile( Threat_Rgn + '-b')

    #probably redo forearm in 256
    Dm_dict = {"hip":256, "knee":200,"foot":200, "torso":225,"forearm":300,
               "upperarm":256, "calf":200, "groin":300,"upperchest":300,"myachingback":250,
               "uchback":300}
    #-1 signifies there is just one zone
    #myachingback changed in 92 from 300 to 250

    Dm1 = Dm_dict[Threat_Rgn]      #2nd dime height in pixels 
    Dm2 = Dm1    #it likes squares
   
    get_train_test_file_list() ##

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

    
    #enables shuffle within epochs

    MODEL_NAME = ('{}x-tz-{}'.format('v71', Dm1, Threat_Rgn ))

    shuffle(TRAIN_SET_FILE_LIST)
    print ('After Shuffling ->', TRAIN_SET_FILE_LIST)

    trainer(Threat_Rgn)  #creates  global var AllTrainingPredicts (#of Views, 5) PZID ThreatLabel MaxView1 MaxView2 MaxView3 

    viewCombiner(Threat_Rgn)




elif control==17:
    
    print("recombine. first save the older")
    PREDICT_PA2 = PATH1 +'/DNN/Predict'
    
    for Threat_Rgn in ['upperarm','forearm','groin','upperchest', ]: #'myachingback',
        #hip 'foot',  'groin',  'torso', 'uchback' calf
        
        print(Threat_Rgn,"_____________")

        AllTrainingPredicts=np.loadtxt(PREDICT_PA2 + "/composite_"+ Threat_Rgn +"_.csv",  delimiter="|")         #5 column format
        HiddenPredictions = np.loadtxt(PREDICT_PA2 + "/compositeH_"+ Threat_Rgn +"_.csv", delimiter="|")         #two column format
        
        viewCombiner(Threat_Rgn)

 