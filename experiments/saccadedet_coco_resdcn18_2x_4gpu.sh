cd src
# train
python main.py saccadedet --exp_id coco_resdcn18 --arch resdcn_18  --batch_size 32 --master_batch 7  --lr 1.25e-4 --gpus 4,5,6,7 --num_workers 16 --num_epochs 230 --lr_step 180,210
# test
python test.py saccadedet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume
# flip test
python test.py saccadedet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test
# multi scale test
python test.py saccadedet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
