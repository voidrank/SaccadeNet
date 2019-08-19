cd src
# train
python main.py saccadedet --exp_id coco_dla_2x_trainval --batch_size 32 --master_batch 7 --lr 1.25e-4 --gpus 4,5,6,7 --num_workers 16 --trainval  --num_epochs 230 --lr_step 180,210
cd ..
