"""Generate from a reference portrait and driving speech."""
import argparse
from protalk.runtime import str2bool


def main():
    args = argparse.ArgumentParser()
    args.add_argument('--hparams', default='configs/model.yaml')
    args.add_argument('--vae_weight', default='weights/retrained/vqvae/best-inference.pth')
    args.add_argument('--sampling_weight', default='weights/retrained/sampler/best-inference.pth')
    args.add_argument('--ref_img', type=str, required=True)
    args.add_argument('--driven_audio', type=str, required=True)
    args.add_argument('--save_dir', type=str, default='results')
    args.add_argument('--model_weight', type=str, default='weights/retrained/expression/best-inference.pth')
    args.add_argument('--pirender_weight', type=str, default='weights/pirender.pt')
    args.add_argument('--mfcc_mean_std_root', type=str, default='data/prepared/mean_std')
    args.add_argument('--pose_scale', type=float, default=None)
    args.add_argument('--device', type=int, default=0)

    args.add_argument('--name', type=str, default='face_recon', help='name of the experiment. It decides where to store samples and models')
    args.add_argument('--model', type=str, default='facerecon', help='chooses which model to use.')
    args.add_argument('--checkpoints_dir', type=str, default='assets/reconstruction')
    args.add_argument('--bfm_folder', type=str, default='assets/bfm')
    args.add_argument('--isTrain', type=str2bool, default=False)
    args.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')
    args.add_argument('--focal', type=float, default=1015.)
    args.add_argument('--center', type=float, default=112.)
    args.add_argument('--camera_d', type=float, default=10.)
    args.add_argument('--z_near', type=float, default=5.)
    args.add_argument('--z_far', type=float, default=15.)
    args.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    args.add_argument('--use_opengl', type=str2bool, nargs='?', const=True, default=False, help='use opengl context or not')
    args.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'], help='network structure')
    args.add_argument('--init_path', type=str, default='assets/reconstruction/resnet50-0676ba61.pth')
    args.add_argument('--use_last_fc', type=str2bool, default=False)
    args.add_argument('--epoch', type=str, default='20', help='which epoch to load? set to latest to use latest cached model')
    args.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    args.add_argument('--use_ddp', type=str2bool, nargs='?', const=True, default=False, help='whether use distributed data parallel')
    parser = args.parse_args()
    ref_data = {
        'ref_img': parser.ref_img,
        'ref_audio': parser.driven_audio,
    }
    from .generate import inference
    inference(opts=(parser, ref_data, parser.device))


if __name__ == "__main__":
    main()
