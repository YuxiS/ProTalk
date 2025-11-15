source /remote-home/yfsong/.bashrc
conda activate ProTalk
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12324 \
 /remote-home/yfsong/code/ProTalk/train_style.py \
 --hparams="/remote-home/yfsong/code/ProTalk/hparams.yaml" \
 --distributed_run=True \
#  --debug True \
# cd /home/songyifei9