cd src
# train
python main.py saccadedet --exp_id coco_dla_1x_aggattv2 --batch_size 32 --master_batch 7 --lr 1.25e-4 --gpus 0,1,2,3 --num_workers 16 --aggatt_mode v2
# test
python test.py saccadedet --exp_id coco_dla_1x_aggattv2 --keep_res --resume --aggatt_mode v2
# flip test
python test.py saccadedet --exp_id coco_dla_1x_aggattv2 --keep_res --resume --flip_test --aggatt_mode v2
# multi scale test
python test.py saccadedet --exp_id coco_dla_1x_aggattv2 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5 --aggatt_mode v2
cd ..
