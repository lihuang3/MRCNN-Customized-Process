######### run bash Edited data: 11/5/2018#############

#### Set up dependencies and build model
cd ./MasK_RCNN
pip install -r requirements.txt --user
python setup.py install --user   										   # need to do this every time if model.py has changed

#### Prepare dataset
cd ./samples/lung
python3 mold_and_prepare_script.py -w ./Mask_RCNN/datasets/lung/ac1 -a 1   # set interval = 1 for adding adjacent channel 

#### Run testing demo
python3 coco_lung_molded.py detect \
        --dataset=../../dataset/liver/ac1 \
        --subset=test \
        --weights=weights/mask_rcnn_nucleus_0040[train]lung_ac1.h5 \
        2>&1 | tee logs/log[test]coco_lung_ac1.txt

#### Run Trainging
python3 coco_lung_molded.py train \
        --dataset=../../datasets/lung/ac1 \
        --subset=train \
        --weights=coco \
        --logs=../../checkpoints/ac1 \
        2>&1 | tee logs/log[train]coco_lung_ac_1.txt               

#### Run Trainging                                                       # weights need to be generated after training  
python3 coco_lung_molded.py detect \
        --dataset=../../datasets/lung/ac1 \
        --subset=test \
        --weights=../../checkpoints/ac1/*/mask_rcnn_nucleus_0040.h5 \
        2>&1 | tee logs/log[test]coco_lung_ac1_new.txt
