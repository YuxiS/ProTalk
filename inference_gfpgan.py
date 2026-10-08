import argparse
import cv2
from glob import glob
import numpy as np
import os
import torch
import av
from fractions import Fraction
from typing import Optional
import pdb
from basicsr.utils import imwrite
from tqdm import tqdm

from gfpgan import GFPGANer
# torch.cuda.set_device(3)



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

def main():
    """Inference demo for GFPGAN (for users).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='results/gfpgan_input',
        help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='temp', help='Output folder. Default: results')
    # we use version to select models, which is more user-friendly
    parser.add_argument(
        '-v', '--version', type=str, default='1.4', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    parser.add_argument(
        '-s', '--upscale', type=int, default=1, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    videos = glob(os.path.join(args.input, '*.mp4'))
    # videos = glob(os.path.join(args.input, '*.avi'))
        # if args.input.endswith('/'):
        #     args.input = args.input[:-1]
        # if args.input.endswith('mp4'):
    for v in tqdm(videos): 
        reader = cv2.VideoCapture(v)
        img_list = []
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            img_list.append(frame)

        # elif os.path.isfile(args.input):
        #     img_list = [args.input]
        # else:
        #     img_list = sorted(glob.glob(os.path.join(args.input, '*')))

        os.makedirs(args.output, exist_ok=True)

        # ------------------------ set up background upsampler ------------------------
        if args.bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                            'If you really want to use it, please modify the corresponding codes.')
                bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=args.bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            bg_upsampler = None

        # ------------------------ set up GFPGAN restorer ------------------------
        if args.version == '1':
            arch = 'original'
            channel_multiplier = 1
            model_name = 'GFPGANv1'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif args.version == '1.2':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANCleanv1-NoCE-C2'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        elif args.version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif args.version == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif args.version == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            raise ValueError(f'Wrong model version {args.version}.')

        # determine model paths
        model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = url

        restorer = GFPGANer(
            model_path=model_path,
            upscale=args.upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)

        # ------------------------ restore ------------------------
        # for img_path in img_list:
        #     # read image
        #     img_name = os.path.basename(img_path)
        #     print(f'Processing {img_name} ...')
        #     basename, ext = os.path.splitext(img_name)
        #     input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        res = []
        for input_img in img_list:
            # restore faces and background if necessary
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                input_img,
                has_aligned=args.aligned,
                only_center_face=args.only_center_face,
                paste_back=True,
                weight=args.weight)
            img = restored_img[:, :, ::-1]
            res.append(img)
        res = np.stack(res, axis=0)
        # pdb.set_trace()
        basename = os.path.split(v)[1][:-3]+'mp4'
        video_path = os.path.join(args.output, basename)
        write_video(video_path, res)
        # save faces
        # for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        #     # save cropped face
        #     save_crop_path = os.path.join(args.output, 'cropped_faces', f'{basename}_{idx:02d}.png')
        #     imwrite(cropped_face, save_crop_path)
        #     # save restored face
        #     if args.suffix is not None:
        #         save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
        #     else:
        #         save_face_name = f'{basename}_{idx:02d}.png'
        #     save_restore_path = os.path.join(args.output, 'restored_faces', save_face_name)
        #     imwrite(restored_face, save_restore_path)
        #     # save comparison image
        #     cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
        #     imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        # if restored_img is not None:
        #     if args.ext == 'auto':
        #         extension = ext[1:]
        #     else:
        #         extension = args.ext

        #     if args.suffix is not None:
        #         save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
        #     else:
        #         save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
        #     imwrite(restored_img, save_restore_path)

    # print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    main()
