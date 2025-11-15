source /remote-home/yfsong/.bashrc
conda activate ProTalk
python reference.py --ref_img $1 --driven_audio $2 --save_dir /remote-home/yfsong/code/ProTalk/test_res
python ./Wav2Lip/inference.py --face /remote-home/yfsong/code/ProTalk/test_res/temp.mp4 --audio $2 --outfile /remote-home/yfsong/code/ProTalk/test_res/temp_wav2lip.mp4 
conda activate FaceRefine  
python ./PSFRGAN/inference_gfpgan.py -i /remote-home/yfsong/code/ProTalk/test_res/temp_wav2lip.mp4 -o /remote-home/yfsong/code/ProTalk/test_res/result.mp4