#!/usr/bin/python

__author__ = 'hcaesar'

# Converts a folder of .png images with segmentation results back
# to the COCO result format. 
#
# The .png images should be indexed images with or without a color
# palette for visualization.
#
# Note that this script only works with image names in COCO 2017
# format (000000000934.jpg). The older format
# (COCO_train2014_000000000934.jpg) is not supported.
#
# See cocoSegmentationToPngDemo.py for the reverse conversion.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

import os, sys, datetime
import io
import json
import re
ROOT_DIR = os.path.abspath('../../')
sys.path.append(os.path.join(ROOT_DIR, "cocoPythonAPI","pycocotools"))


if not os.path.exists(os.path.join(ROOT_DIR, 'coco_results')):
    os.makedirs(os.path.join(ROOT_DIR, 'coco_results'))

from cocostuffhelper import pngToCocoResult



def pngToCocoResultDemo(dataDir=None, labelDir = None):
    '''
    Converts a folder of .png images with segmentation results back
    to the COCO result format. 
    :param dataDir: location of the COCO root folder
    :param resType: identifier of the result annotation file
    :param indent: number of whitespaces used for JSON indentation
    :return: None
    '''

    # Define paths
    imgFolder = '%s' % (dataDir)
    jsonPath = '%s'%(ROOT_DIR) +'/coco_results/segm_results.json'

    # Get images in png folder
    imgNames = os.listdir(imgFolder)

    # imgNames = [imgName[:-4] for imgName in imgNames if imgName.endswith('.tif')]
    imgNames.sort()
    imgCount = len(imgNames)

    # Init
    annCount = 0

    with io.open(jsonPath, 'w', encoding='utf8') as output:
        print('Writing results to: %s' % jsonPath)
        # Evalset image info start
        
        # Annotation start
        output.write(str('[\n'))

        for i, imgName in zip(range(0, imgCount), imgNames):
            print('Converting png image %d of %d: %s' % (i+1, imgCount, imgName))

            # Add stuff annotations
            labelName = 'seg'+imgName
            labelPath = '%s/%s.tif' % (labelDir, labelName)        
            imgId = int(imgName.replace("-","") )

   
            
            anns = pngToCocoResult(labelPath, imgId, stuffStartId=0)
            # Process for json.dumps(anns) in Python 3.x 
            for item in anns:
                if type(item['segmentation']['counts'])==bytes:
                    item['segmentation']['counts'] = item['segmentation']['counts'].decode('utf-8')

            str_ = json.dumps(anns)
            str_ = str_[1:-1]
            if len(str_) > 0:
                output.write(str(str_))
                annCount = annCount + 1

            # Add comma separator
            if i < imgCount-1 and len(str_) > 0:
                output.write(str(','))

            # Add line break
            output.write(str('\n'))

        # Annotation end
        output.write(str(']'))

        # Create an error if there are no annotations
        if annCount == 0:
            raise Exception('The output file has 0 annotations and will not work with the COCO API!')

if __name__ == "__main__":
    # path to the folder which contains all evaluation segm labels
    segmlabel_folder = os.path.join(ROOT_DIR, 'dataset','liver','seg')
    # path to the folder which contains all test images
    test_folder = os.path.join(ROOT_DIR, 'dataset','liver','ac1','test')
    pngToCocoResultDemo(dataDir = test_folder, labelDir = segmlabel_folder)
