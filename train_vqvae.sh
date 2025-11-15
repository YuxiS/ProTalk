source /remote-home/yfsong/.bashrc
conda activate ProTalk
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12235 \
 /remote-home/yfsong/code/ProTalk/train_script/train_vqvae.py \
 --hparams="/remote-home/yfsong/code/ProTalk/hparams.yaml" \
 --distributed_run True --save_dir /remote-home/yfsong/code/ProTalk/weights/vqvae \
 --debug False