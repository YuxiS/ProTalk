from os import listdir, path
import numpy as np
import cv2, os, sys, argparse, audio
from tqdm import tqdm
from glob import glob
import torch
from models import Wav2Lip
from multiprocessing import Pool
from itertools import cycle
import pdb
from fractions import Fraction
from typing import Optional
import av


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/wav2lip.pth',
					help='Name of saved checkpoint to load weights from')

parser.add_argument('--face', type=str, default='./results/1059_ITS_NEU_XX.mp4',
					help='Filepath of video/image that contains faces to use')
parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', default='./results/test.wav',)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=30., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=2)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=32)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True


def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def insertbox(boxes):
	# boxes :list [[x, y ,x_1, y_1]]
	for i in range(len(boxes)):
		if boxes[i][0]==-1:
			if i==0:
				right = boxes[i]
				for j in range(i+1, len(boxes)):
					if boxes[j][0]!=-1:
						right = boxes[j]
						break
				if right[0]==-1:
					return None
				for k in range(4):
					boxes[i][k]=right[k]
			else:
				left = boxes[i-1]
				right = boxes[i]
				for j in range(i+1, len(boxes)):
					if boxes[j][0]!=-1:
						right = boxes[j]
						break 
				if right[0]!=-1:
					for k in range(4):
						boxes[i][k]=int((right[k]+left[k])/2)
				else:
					for k in range(4):
						boxes[i][k]=left[k]
		return boxes
				
			
def face_detect(images, video_file):
	predictions = []
	face_detector = cv2.CascadeClassifier('./checkpoints/haarcascade_frontalface_default.xml')
	for img in images:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_detector.detectMultiScale(gray, 1.3, 5)
		if len(faces)>0:
			x, y ,w ,h = faces[0]
		else:
			if len(predictions)==0:
				x, y, w, h = -1, -1, 0 ,0
		predictions.append([x, y, x+w, y+h])		
	predictions = insertbox(predictions)	
	if predictions is None:
			raise ValueError("No Face in Video in {}".format(video_file))
	
	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	# del detector
	return results 

def datagen(frames, mels, video_file):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames, video_file) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]], video_file)
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

	for i, m in enumerate(mels):
		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0
		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path, device):
	model = Wav2Lip()
	# print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()
def write_video(
    output_fn: str,
    frames: np.array,
    sample_rate: Optional[int] = 30,
    output_codec: Optional[str] = "h264",
    output_options: Optional[dict] = {},
    pix_fmt: Optional[str] = "yuv420p",
    frame_format: Optional[str] = "rgb24",
) -> None:
    """
    write video with pyav

    output_fn: The output file name, e.g. 'output.mp4'
    frames: The frames to write to the output file, expected to be 4D numpy array in the format (t, h, w, c)
    sample_rate: The sample rate of the output video, default to 30
    output_codec: The output codec, default to 'h264', use `ffmpeg -codecs` to list all available codecs
    output_options: The output options, default to {}
    pix_fmt: The pixel format of the output video, default to 'yuv420p', use `ffmpeg -pix_fmts` to list all available formats
    frame_format: The format of the input frames, default to 'rgb24', use `ffmpeg -pix_fmts` to list all available formats
    """
    # Open the output file
    container = av.open(output_fn, "w")

    # Set the output format to H.264
    video_stream = container.add_stream(output_codec, options=output_options)
    video_stream.pix_fmt = pix_fmt

    # Set the frame rate and frame size
    video_stream.rate = sample_rate
    video_stream.width = frames.shape[2]
    video_stream.height = frames.shape[1]
    video_stream.time_base = Fraction(1, sample_rate)

    # Write the frames to the output file
    for frame in frames:
        # Encode and write the video frame
        video_frame = av.VideoFrame.from_ndarray(frame, format=frame_format)
        for packet in video_stream.encode(video_frame):
            container.mux(packet)

    # Flush the encoders
    for packet in video_stream.encode():
        container.mux(packet)

    # Close the output file
    container.close()

def generate(opts):
	wav_file, video_file, out_file, device = opts
	# pdb.set_trace()
	model = load_model(args.checkpoint_path, device)
	if not os.path.isfile(video_file):
		raise ValueError('--face argument must be a valid path to video/image file')

	elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args.face)]
		fps = args.fps
	else:
		video_stream = cv2.VideoCapture(video_file)
		fps = video_stream.get(cv2.CAP_PROP_FPS)

		print('Reading video frames...')

		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()
			if not still_reading:
				video_stream.release()
				break
			if args.resize_factor > 1:
				frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

			if args.rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args.crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

	wav = audio.load_wav(wav_file, 16000)
	mel = audio.melspectrogram(wav)
	# print(mel.shape)

	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

	mel_chunks = []
	mel_idx_multiplier = 80./fps 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1
	full_frames = full_frames[:len(mel_chunks)]

	batch_size = args.wav2lip_batch_size
	gen = datagen(full_frames.copy(), mel_chunks, video_file)
	res_frames = []
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			frame_h, frame_w = full_frames[0].shape[:-1]

		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, img_batch)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
			res_frames.append(np.array(f))
	res_frames = np.stack(res_frames, axis=0)
	write_video(out_file, res_frames, sample_rate=int(fps))

if __name__=='__main__':
	# VIDEOS_ROOT = '/remote-home/yfsong/code/TalkingHead/MakeItTalk-main/examples/CREMA-D'
	# VIDEOS_ROOT = '/remote-home/yfsong/code/TalkingHead/MakeItTalk-main/examples/RAVEDSS'
	# WAV_ROOT = '/remote-home/yfsong/code/TalkingHead/MakeItTalk-main/examples'
	# METHOD = 'MakeItTalk'
	##################################################################################
	# VIDEOS_ROOT = '/remote-home/yfsong/code/TalkingHead/Audio2Head-main/results/CREAM-D'
	# VIDEOS_ROOT = '/remote-home/yfsong/code/TalkingHead/Audio2Head-main/results/RAVEDSS'
	# WAV_ROOT = '/remote-home/share/yfsong/CREMA-D/wav'
	# WAV_ROOT = '/remote-home/share/yfsong/RAVEDSS/wav'
	# METHOD = 'Audio2Head'
	#################################################################################
	# VIDEOS_ROOT = '/remote-home/yfsong/code/TalkingHead/AAAI22-one-shot-talking-face-main/samples/res/CREMA-D'
	# VIDEOS_ROOT = '/remote-home/yfsong/code/TalkingHead/AAAI22-one-shot-talking-face-main/samples/res/RAVEDSS'
	# WAV_ROOT = '/remote-home/share/yfsong/CREMA-D/wav'
	# WAV_ROOT = '/remote-home/share/yfsong/RAVEDSS/wav'
	# METHOD = 'OneShot'
	#################################################################################
	# VIDEOS_ROOT = '/remote-home/yfsong/code/TalkingHead/Real3DPortrait-main/infer_out/CREMA_D'
	VIDEOS_ROOT = '/remote-home/yfsong/code/TalkingHead/Real3DPortrait-main/infer_out/RAVEDSS'
	# WAV_ROOT = '/remote-home/share/yfsong/CREMA-D/wav'
	WAV_ROOT = '/remote-home/share/yfsong/RAVEDSS/wav_resample'
	METHOD = 'Real3D'
	

	# save_dir = os.path.join('./results/TalkingHeadsVideos/', METHOD, 'CREMA-D')
	save_dir = os.path.join('./results/TalkingHeadsVideos/', METHOD, 'RAVEDSS')
	torch.multiprocessing.set_start_method('spawn', force=True)
	os.makedirs(save_dir, exist_ok=True)
	test_videos = glob(os.path.join(VIDEOS_ROOT, "*.mp4"))
	ref_data = []
	# pool = Pool(processes=1)
	torch.cuda.set_device(1)
	pdb.set_trace()
	# opts = cycle([parser])
	# devices = cycle(['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
	for i, v in tqdm(enumerate(test_videos)):
		basename= os.path.basename(v)
		# wav_file =os.path.join(WAV_ROOT, 'Actor_'+str(os.path.splitext(basename)[0].split('-')[-1])+'-'+basename.replace('.mp4', '.wav')) # RAVEDSS
		# wav_file =os.path.join(WAV_ROOT, basename.replace('.mp4', '.wav'))
		# wav_file = os.path.join(WAV_ROOT, '_'.join(basename.split('_')[-4:]).replace('.mp4', '.wav')) # CREMA-D
		# wav_file = os.path.join(WAV_ROOT, basename.split('-')[-1].replace('.mp4', '.wav'))
		# wav_file = os.path.join(WAV_ROOT,  'Actor_'+str(basename.split('_')[-1].split('-')[-1][:2])+'-'+basename.split('_')[-1].replace('.mp4', '.wav'))
		wav_file = os.path.join(WAV_ROOT, '-'.join(basename.split('-')[1:]).replace('.mp4', '.wav'))
		if not os.path.exists(wav_file):
			continue
		out_file = os.path.join(save_dir, basename)
		ref_data.append(
			{
				'video': v,
				'audio': wav_file,
				'outfile': out_file
			}
		)
		generate((wav_file, v, out_file, 'cuda:{}'.format(3)))
	# 	pool.imap_unordered(generate, zip(wav_file, v, out_file, devices))
	# pool.close()