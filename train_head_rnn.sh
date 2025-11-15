source /remote-home/yfsong/.bashrc
conda activate ProTalk
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12365 \
 /remote-home/yfsong/code/ProTalk/train_script/train_head_rnn.py \
 --hparams="/remote-home/yfsong/code/ProTalk/hparams.yaml" \
 --distributed_run True \
 --save_dir /remote-home/yfsong/code/ProTalk/weights/ablia/headrnn --batch_size 128 \
#  --debug True