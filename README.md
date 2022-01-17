## Get Started

- Check `datasets/MultiFDataset.py` to generate GT files for training. Note that you should prepare `k400_mr345_*SPLIT*_min_change_duration0.3.pkl` for Kinetics-GEBD and `TAPOS_*SPLIT*_anno.pkl`  (this should be organized as `k400_mr345_*SPLIT*_min_change_duration0.3.pkl` for convenience) for TAPOS before running our code. 

-  Accordingly change `PATH_TO` in our codes to your data/frames path as needed.

- Train on Kinetics-GEBD:

 ```shell
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 10078 PC_train_cascade.py \
--dataset kinetics_multiframes \
--train-split train \
--val-split val \
--num-classes 2 \
--batch-size 16 \
--n-sample-classes 2 \
--n-samples 8 \
--lr 0.01 \
--warmup-epochs 0 \
--clip-grad 1 \
--epochs 30 \
--decay-epochs 2 \
--model mc3lstm_cascade \
--pin-memory \
--balance-batch \
--native-amp \
--eval-metric F1 \
--log-interval 50 \
--exp dexiang_dynamic_self_att_global_csn \
--amp \
--min_change_dur_list 0.5 0.4 0.3 \
--filter_thresh 0 0 \
--use_train_val \
--use_min_val \
--frame_per_side 8
 ```

- Generate scores sequence on Kinetics-GEBD Validation Set:

  ```shell
  python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 10078  \ 
  --model mc3lstm_cascade_audio \
  --resume path_to/checkpoint
  ```
  
  ## Model Zoo
  链接: https://pan.baidu.com/s/1IOAAKeZkHmurXmtAnVrl8A  密码: 31l1

