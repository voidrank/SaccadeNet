cd src
# train
python main.py saccadedet --exp_id coco_dla_1x_ctreg --batch_size 32 --master_batch 7 --lr 1.25e-4 --gpus 0,1,2,3 --use_agg_att_ctreg --num_workers 16
# test
python test.py saccadedet --exp_id coco_dla_1x_ctreg --keep_res --resume  --use_agg_att_ctreg
# flip test
python test.py saccadedet --exp_id coco_dla_1x_ctreg --keep_res --resume --flip_test  --use_agg_att_ctreg
# multi scale test
python test.py saccadedet --exp_id coco_dla_1x_ctreg --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5  --use_agg_att_ctreg
cd ..
