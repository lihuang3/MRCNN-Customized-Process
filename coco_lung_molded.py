''' Lung detection 
Demo: 
    [Test use pretrained weights]                                   # weights in the files, ready to use
        python3 nucleus_lung_molded.py detect \
        --dataset=../../datasets/lung/ac1 \
        --subset=test \
        --weights=weights/mask_rcnn_nucleus_0040[train]lung_ac1.h5 \
        2>&1 | tee logs/log[test]coco_lung_ac1.txt

Experiment
    [Train]                                                          #  specify path of checkpoints and trained weights to ../../checkpoints/ac1
        python3 coco_lung_molded.py train \
        --dataset=../../datasets/lung/ac1 \
        --subset=train \
        --weights=coco \
        --logs=../../checkpoints/ac1 \
        2>&1 | tee logs/log[train]coco_lung_ac_1.txt               

    [Test]                                                         # weights need to be generated after training  
        python3 coco_lung_molded.py detect \
        --dataset=../../datasets/lung/ac1 \
        --subset=test \
        --weights=../../checkpoints/ac1/*/mask_rcnn_nucleus_0040.h5 \
        2>&1 | tee logs/log[test]coco_lung_ac1_new.txt
'''

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import warnings
warnings.filterwarnings("ignore")
import cv2
import random
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
import colorsys
import time 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize


sys.path.append(os.path.join(ROOT_DIR, "cocoPythonAPI","pycocotools"))

from cocoPythonAPI.cocostuff import *
from cocoPythonAPI.pycocotools import cocostuffhelper
from cocoPythonAPI.pycocotools.coco import COCO
from cocoPythonAPI.pycocotools.cocostuffeval import COCOStuffeval
from cocoPythonAPI.pycocotools.cocoeval import COCOeval
from cocoPythonAPI.pycocotools import mask as maskUtils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")                 # save the checkpoints and weights

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/lung/")

############################################################
#  Configurations
############################################################

class NucleusConfig(Config):

    def __init__(self, dataset):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES

        self.dataset = dataset
        print ("!!!!" + dataset)

    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "liver"
    LEARNING_RATE = 0.0001

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 8   # default = 6

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1  # Background + class 1 + class2

    # VAL_IMAGE_IDS = []


    # # Number of training and validation steps per epoch
    # STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    # VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # trainset_size = len(os.listdir(os.path.join(self.dataset, "train")))
    # evalset_size =  len(os.listdir(os.path.join(self.dataset, "test")))
    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 8000 // IMAGES_PER_GPU
    VALIDATION_STEPS = 400 // IMAGES_PER_GPU

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 0.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class NucleusDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("nucleus", 1, "Gray")
        self.add_class("nucleus", 2, "White")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        
        #     assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        # assert subset in ["train", "val", "test","wholeBrain_imadjusted_overlapped"]
        subset_dir = subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)

        image_ids = next(os.walk(dataset_dir))[1]

        # Add images
        image_ext = os.listdir(os.path.join(dataset_dir, image_ids[0], "images"))[0].split(".")[1]
        for image_id in image_ids:
            # print (image_id)          
            self.add_image(
                "nucleus",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.".format(image_id) + image_ext))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m   = skimage.io.imread(os.path.join(mask_dir, f))
                m_bin = m.astype(np.bool)    
                mask.append(m_bin)
                m_class = 0
                if len(np.unique(m) > 1):                                                   # at least one objected is detected
                    m_class = int(np.unique(m).max()/127)                                   # in 0: background, : Class1 ,255 class2                     
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32) * m_class

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleusDataset()
    dataset_train.load_nucleus(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(dataset_dir, "test")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    # print("Train network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=50,
    #             augmentation=augmentation,
    #             layers='heads'
    #             )

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=200,
                augmentation=augmentation,
                layers='all' )


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def proofreading(r):   # Rebecca    
    # base on descent order of r['scores']
    detected_classes = np.unique(r["class_ids"])          # [1] or [1 2]
    take_ids = []
    for detected_class in  detected_classes:
        detected_ids = np.where(r["class_ids"] == detected_class)
        take_ids.append( np.array(detected_ids,dtype = int).min())
    take_ids = np.array(take_ids,dtype = int)
    # print (take_ids)
    r_new = {}
    r_new['rois']       = r['rois'][take_ids,:]
    r_new['masks']      = r['masks'][:,:,take_ids]
    r_new['class_ids']  = r['class_ids'][take_ids]
    r_new['scores']     = r['scores'][take_ids]

    return r_new

def gray2color(img_2d):
    img_3d = np.stack ((img_2d,img_2d,img_2d), axis =2) 
    return img_3d

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]


    auto_show = False


    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]


    masked_image = image.astype(np.uint8).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        show_bbox = False
        if show_bbox:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        
        ret,thresh = cv2.threshold(padded_mask,127,255,0)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(masked_image, contours, -1, (0,0,255), 2)

    return masked_image


def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    
    submission = []

    eval_len = len(dataset.image_ids)    
    for i, image_id in enumerate(dataset.image_ids):
                    
        print('\rRunning process on %d of %d images, %s ...'%(i+1, eval_len, str(image_id)), end="" )
        # # Load image and run detection
        image = dataset.load_image(image_id)
        # # Detect objects
        r = model.detect([image], verbose=0)[0]
        # # Encode image to RLE. Returns a string of multiple lines
        
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"]) 
        submission.append(rle)                   # save all detected objects, write into file

        # Save image with masks
        r = proofreading(r)                     # Edited : only maintain highest score one in each class
        image = image[:,:,1]                    # Edited:  only extract midchannel 
        image = gray2color(image)  

        # visualize.display_instances(
        #     image, r['rois'], r['masks'], r['class_ids'],
        #     dataset.class_names, r['scores'],
        #     show_bbox=False, show_mask=True,
        #     title= ( "Predictions on " + dataset.image_info[image_id]["id"]))

 
        masked_image = display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'])
        cv2.imwrite('%s/%s.png'%(submit_dir, dataset.image_info[image_id]["id"]), masked_image) 

        # plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []

    for image_id in image_ids:
        
        if rois.shape[0] == 0:
            class_id = 0
            score = 0
            bbox = [0,0,512, 512]
            mask = np.zeros([512, 512], dtype= np.uint8)

            result = {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
            continue
                        
        # Loop through detections

        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]
            
            result = {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)

    return results

def evaluate_coco(model, dataset_dir, subset = "test", annFile = None, eval_type="segm", limit=0):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Validation dataset
    dataset_val = NucleusDataset()
    dataset_val.load_nucleus(dataset_dir, subset)
    dataset_val.prepare()

    # Initialize COCO ground-truth API
    cocoGt = COCO(annFile)    

    # Pick COCO images from the dataset
    image_ids = dataset_val.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.


    coco_image_ids = [dataset_val.image_info[id]["id"] for id in image_ids]
    coco_image_ids = [int(imgId.replace("-","")) for imgId in coco_image_ids]

    t_prediction = 0
    t_start = time.time()
    
    val_len = len(image_ids)    

    results = []
    for i, image_id in enumerate(image_ids):
        print('\rRunning process on %d of %d images ...'%(i+1, val_len), end="" )

        # Load image
        image = dataset_val.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool

        image_results = build_coco_results(dataset_val, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    cocoRes = cocoGt.loadRes(results)

    # Evaluate
    cocoEval = COCOStuffeval(cocoGt, cocoRes)
    before = time.clock()
    # Run evaluation on the example images
    cocoEval.evaluate()
    cocoEval.summarize()
    # Print time
    after = time.clock()
    print('Evaluation took %.2fs!' % (after - before))

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Command Line
############################################################

ROOT_DIR = os.path.abspath('../../')


if __name__ == '__main__':

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        default = ROOT_DIR+"/dataset/liver/ac1",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        default = "coco",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",                        
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        default = "test",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)

    if args.subset:
        print("Subset: ", args.subset)

    print("check points and  weights save as: (args.logs) ", args.logs)
    
    # Configurations
    if args.command == "train":
        VAL_IMAGE_IDS  = os.listdir(os.path.join(args.dataset, "train"))[0:10]
        NucleusConfig.VAL_IMAGE_IDS = VAL_IMAGE_IDS
        config = NucleusConfig(args.dataset)

    else:
        config = NucleusInferenceConfig(args.dataset)
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", 
                                  config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", 
                                  config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if "coco" in args.weights.lower():
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == "eval":
        groudtruth = os.path.join(ROOT_DIR,  'coco_results','coco_segm.json')
        print("Running COCO evaluation on the validation set")

        evaluate_coco(model, args.dataset, annFile=groudtruth, limit = 0)        

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))