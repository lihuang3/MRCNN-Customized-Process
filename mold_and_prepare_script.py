''' Lung detection Data preparation
Oct, 2018

[e.g]
    python3 mold_and_prepare_script.py -h                                      # check the detail of the function

[Write in to molded folder: ac1]                                               # tend to have better result
    python3 mold_and_prepare_script.py -w ./Mask_RCNN/datasets/lung/ac1 -a 1   # set interval = 1 for adjacent channel consideration

[Write in to molded folder: ac2]                                               # to be test ( ac = 3 is worse than ac = 1)
    python3 mold_and_prepare_script.py -w ./Mask_RCNN/datasets/lung/ac2 -a 2   # set interval = 2 for adjacent channel consideration
          
'''
import sys,os
import matplotlib as mpl
mpl.use('TkAgg')
import skimage.draw
import skimage.io as io
import skimage.measure 
from skimage.filters import try_all_threshold
import shutil
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import util,segmentation,exposure,filters, morphology,measure,feature,io,draw
from scipy import ndimage,stats,cluster,misc,spatial
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
import warnings
warnings.filterwarnings("ignore")

def remove_emptyMask_sample (sample_root):                                           # make sure no emply objects         
    for image_ID in os.listdir(sample_root):
        sample_dir = os.path.join (sample_root,image_ID)
        masks_dir = os.path.join (sample_dir,"masks")
        
        if (os.path.exists(masks_dir) == False) or (len(os.listdir(masks_dir)) ==0) :
            print ("REMOVE sample", sample_dir)
            shutil.rmtree(sample_dir)


def checkPath(Path):
    if os.path.exists(Path) is False:                                               # Create current directory if needs
        os.makedirs(Path)
    return Path

def gray2color(img_2d):
    img_3d = np.stack ((img_2d,img_2d,img_2d), axis =2) 
    return img_3d

def adjacentChannel(vol_img, imageID,img_dir, channelwidth = 1):
    vol_img_ac = gray2color(vol_img)                                                # vol image by adding adjacent channel  
    
    # Add the left and right channel
    
    # imageID_prefix = imageID.split("-")[-1]
    # imageID_left   = imageID.split(imageID_prefix)[0] + str(int(imageID_prefix) - channelwidth)
    # imageID_right  = imageID.split(imageID_prefix)[0] + str(int(imageID_prefix) + channelwidth)        
    
    imageID_elem    = imageID.split("-")
    imageID_left    = imageID_elem[0] + "=" + str(int(imageID_elem[-1]) - channelwidth)
    imageID_right   = imageID_elem[0] + "=" + str(int(imageID_elem[-1]) + channelwidth)

    vol_img_FileName_left  = "vol" +  imageID_left  + ".tif"
    vol_img_FileName_right = "vol" +  imageID_right + ".tif"        
    if os.path.isfile(os.path.join( img_dir  , vol_img_FileName_left) ) :
        vol_img_ac [:,:,0] = io.imread( os.path.join( img_dir  , vol_img_FileName_left) )  
    if os.path.isfile(os.path.join( img_dir  , vol_img_FileName_right) ) :
        vol_img_ac [:,:,2] = io.imread( os.path.join( img_dir  , vol_img_FileName_right) )   
        
    return vol_img_ac

'''Devide Training and testing'''
def split_train_test ( samples_folder, write_folder, test_rate = 0.10, interval = 1, remove_input= False):
    ''' Select sequenced samples rather than independent samples'''
    remove_emptyMask_sample (samples_folder)                    

    temp_dir        = samples_folder
    print (temp_dir)
    arr = [ id  for id in range(0, len( os.listdir(temp_dir)), interval)]
    arr_shuffled = arr[0:-1]
    np.random.seed(seed = 300 )                                                              # just to make sure the same testing set is selected every time
    np.random.shuffle(arr_shuffled)
    test_num        = int(len( os.listdir(temp_dir))* test_rate)                             # number of testing samples
    squence_numbers = 1 if int(test_num/interval) < 1 else int(test_num/interval)
    test_ids_start  = np.array( arr_shuffled[0:squence_numbers]  )                           # testing sequence starting ids
    print ("test_num =", test_num, "test_ids_start = ", test_ids_start) 
    train_ids =  list( range(0, len( os.listdir(temp_dir))) )                                # list for training sample ids
    test_dir  = checkPath(os.path.join (write_folder,"test"))
    train_dir = checkPath(os.path.join (write_folder,"train"))    
    # for id, sample in enumerate( os.listdir(temp_dir) ):
    #     for test_id_start in test_ids_start:
    #         if id >= test_id_start and id < test_id_start + interval:
    #             print (id)
    #             train_ids.remove(id)                                                          # remove the testing samples ids from training sample list
    #             shutil.move( os.path.join(temp_dir,sample ) ,
    #                         os.path.join (test_dir, sample)
    #                         )
    # for id, sample in enumerate( os.listdir(temp_dir) ):
    #     if id in train_ids:
    #         shutil.move( os.path.join(temp_dir,sample ) ,
    #                     os.path.join (train_dir, sample)
    #                     ) 
    for id, sample in enumerate( os.listdir(temp_dir) ):
        if id in test_ids_start:
            print (id)
            train_ids.remove(id)                                                          # remove the testing samples ids from training sample list
            shutil.move( os.path.join(temp_dir,sample ) ,
                        os.path.join (test_dir, sample)
                        )
        else:
            shutil.move( os.path.join(temp_dir,sample ) ,
                        os.path.join (train_dir, sample)
                        ) 


    if remove_input== True:
       shutil.rmtree(samples_folder)                         

def mold_and_prepare(root_dir, dataset_dir, write_folder, op ,ac =0):
    if op == "MRCNN":
        molded_dir        = checkPath(  write_folder  )          
        molded_dir_data   = checkPath( os.path.join(molded_dir,"temp") )                    # temporarily save the data
        
        print ("--- Mold samples---")
    #    subset_ls = ["file28", "file30","file33", "file36","file41"]                       # may specify this for train small dataset
        # subset_ls = os.listdir(dataset_dir)   
        # for subset in subset_ls:
        dataset     = dataset_dir  #os.path.join ( dataset_dir, subset)
        label_dir   = os.path.join ( dataset, "seg")                             
        img_dir     = os.path.join ( dataset, "vol")                            
        for label_img_FileName in os.listdir(label_dir):
            if not '.tif' in label_img_FileName:
                continue    
            # Read Label
            imageID          = label_img_FileName.replace("seg", "")
            vol_img_FileName = "vol" +  imageID
            imageID          = imageID.replace(".tif", "")

            label_img   = io.imread( os.path.join( label_dir, label_img_FileName) )     # e.g. seg28-28-10.tif
            vol_img     = io.imread( os.path.join( img_dir  , vol_img_FileName) )      # e.g. vol28-28-10.tif
            
#                print ("+"*20, "\n", imageID)
            
            sample_folder    = os.path.join(molded_dir_data, imageID)
            if os.path.exists(sample_folder) is False:
                os.mkdir(sample_folder)
                    
            # Write Image
            images_subfolder = os.path.join(sample_folder, "images")
            if os.path.exists(images_subfolder) is False:
                os.mkdir(images_subfolder)
            
            if ac > 0 :   # add adjacentChannel
                vol_img_ac = adjacentChannel(vol_img, imageID,img_dir, channelwidth = ac)                
                io.imsave( os.path.join(images_subfolder,imageID + ".tif" ), vol_img_ac )
            else:        
                shutil.copy( os.path.join( img_dir  , vol_img_FileName),
                            os.path.join(images_subfolder,imageID + ".tif" )
                        )    
            
            # Write Single label Imgs
            masks_subfolder  = os.path.join(sample_folder, "masks")    
            if os.path.exists(masks_subfolder) is False:
                os.mkdir(masks_subfolder)          
            labels = skimage.measure.label(label_img)  
            if labels.max() > 0 :                                                               # run only when there is at least one obj in the image 
                labels = skimage.measure.label(label_img)                                       # separate the object with same class ids
                for obj in skimage.measure.regionprops(labels, intensity_image = label_img):
                    if obj.area > 5:                                                            # remove the objects are too small 
                        m = np.zeros( ( label_img.shape[0],label_img.shape[1] ) ,   dtype = np.uint8)
                        m_filled_image = obj.intensity_image                                    # 0: background, 128: Class1 ,255 class2 

                        m [ obj.bbox[0] : obj.bbox[2],
                            obj.bbox[1] : obj.bbox[3]] = m_filled_image
                                                        
                        io.imsave( os.path.join( masks_subfolder,
                                                 imageID + "-" + str (obj.label) + ".png"), m)
            else:
                print ("Warning, their is no object detected in the image " ,vol_img_FileName)
                m = np.zeros( ( label_img.shape[0],label_img.shape[1] ) , 
                                dtype = np.uint8)
                io.imsave( os.path.join( masks_subfolder, imageID + "-0" + ".png"), m)
    
        '''Devide Training and testing'''    
        remove_emptyMask_sample (molded_dir_data)
        print ("        '''Devide Training and testing'''    ")
        split_train_test ( samples_folder = molded_dir_data, write_folder = molded_dir, 
                           test_rate = 0.01, remove_input= True)
        
    elif op == "GAN":
        
        canvas_size =  [256,256]      
    #    subset_ls = ["file28", "file30","file33", "file36","file41"] 
        write_dir   = checkPath(os.path.join(root_dir, write_folder))
            
        sample_folder = checkPath(os.path.join(write_dir, "temp"))
    
        for subset in os.listdir(dataset_dir):
            dataset     = os.path.join ( dataset_dir, subset)
            label_dir   = os.path.join ( dataset, "seg")                             
            img_dir     = os.path.join ( dataset, "volume")                            
            for label_img_FileName in os.listdir(label_dir):                
                # Read Label
                imageID          = label_img_FileName.split("seg")[1] .split(".")[0]
                vol_img_FileName = "vol" +  imageID + ".tif"
    
                label_img   = io.imread( os.path.join( label_dir, label_img_FileName) )    # e.g. seg28-28-10.tif
                vol_img     = io.imread( os.path.join( img_dir  , vol_img_FileName) )    # e.g. vol28-28-10.tif
    
                label_img    = gray2color(label_img)
                if ac >0 :   
                    vol_img = adjacentChannel(vol_img, imageID,img_dir, channelwidth = ac)            
                else:
                    vol_img      = gray2color(vol_img)
    
                # resize the original imaage , mask img to make it exact the same as the canvas
                label_img_resized   = misc.imresize (  label_img,  [canvas_size[0],canvas_size[1],3])
                vol_img_resized     = misc.imresize (  vol_img   , [canvas_size[0],canvas_size[1],3])    
    
                label_img_resized = label_img_resized
        
                GANinput = np.concatenate ( (label_img_resized , vol_img_resized ), axis = 1)  #  canvas_size = > canvas_size *2
                io.imsave(os.path.join(sample_folder,imageID + ".jpg"), GANinput)
    
        '''Devide Training and testing'''
    
        split_train_test ( samples_folder = sample_folder, write_folder = write_dir, test_rate = 0.005, remove_input= False)
           

if __name__ == '__main__':
    import argparse
    
    ''' Input variables'''
    parser = argparse.ArgumentParser()
                        
    parser.add_argument('-r','--root_dir', required=False,
                        default = os.path.abspath("../../"),
                        help='Root path of the project')
    parser.add_argument('-s','--subset', required=False,
                        default = "dataset/liver",
                        help="dataset name under root: Raw image inputs")
    parser.add_argument('-w','--write_folder', required=False,
                        default = None,
                        help="full path of write dir")
    parser.add_argument('-o', '--op', required = False,
                        default = 'MRCNN', type = str,
                        help= " option to molded for Deep learning model : GAN/MRCNN")
    parser.add_argument('-a', '--ac', required = False,
                        default = 1, type = int,
                        help= " Add adjacentChannel info, 0 not add, int >1 add channel interval ")
    args = parser.parse_args()

    if args.root_dir  is None:
        root_dir = os.getcwd()
    else:
        root_dir = args.root_dir    

    if args.write_folder  is None:
        write_folder = os.path.join(root_dir,"dataset","train","ac1")
    else:
        write_folder = args.write_folder  

    dataset_dir  = os.path.join( root_dir, args.subset)    
    
    mold_and_prepare(root_dir, dataset_dir, write_folder,
                     op = args.op, ac = args.ac)
    
    
    
    
    