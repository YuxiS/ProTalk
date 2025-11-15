import torch
from tqdm import tqdm
import sys
sys.path.append('.')
sys.path.append('..')
from face_utils.utils import read_video, write_video
from face_utils.recons import DeepFace3DReconModel, DeepFace3DReconPreprocessor
from face_utils.renders import PIRenderFaceGenerator, PIRenderPreprocessor, PIRenderPostProcessor
import cv2
import numpy as np
from torchvision import transforms
import os
from scipy.io import savemat
import pdb



class ImgDataset:
    def __init__(self, data_list) -> None:
        self.model = DeepFace3DReconModel(
            resume_path='/home/songyifei9/code/prosody/StyleProsody/mellotron/deep3d/checkpoints/face_recon/epoch_20.pth',
            bfm_folder='/home/songyifei9/code/prosody/StyleProsody/mellotron/deep3d/BFM',
            bfm_model='BFM_model_front.mat',
        ).cuda()
        self.recon_preprocessor = DeepFace3DReconPreprocessor(
            bfm_folder='/home/songyifei9/code/prosody/StyleProsody/mellotron/deep3d/BFM',
        ).cuda()

        with open(data_list, mode='r') as f:
            files = f.readlines()
        self.datalist = [l.strip() for l in files]

        self.coeff_root = '/home/songyifei9/data/FFHQ/3dmm_coeffs'
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256)]
        )
    
    def prepare(self, img_path):
        f = open('./broken_imgs.txt',mode='a')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transforms(img)
        img = torch.from_numpy(np.array(image))
        img = img.unsqueeze(0)
        # trans_params, new_img, new_landmarks, new_masks = self.recon_preprocessor.align_and_recrop(img)
        new_img, trans_params = self.recon_preprocessor.image_transform(img, image)
        # pdb.set_trace()
        if trans_params is None:
            f.write(img_path+'\n')
            f.close()
            return None
        f.close()
        new_img = new_img.unsqueeze(0).cuda()
        # new_img = new_img.contiguous().permute(0, 2 ,3 ,1)
        # new_img_tensor = self.recon_preprocessor.prepare_input(new_img)
        with torch.no_grad():
            coeffs = self.model.extract_coeffs(new_img)
        coeffs['trans_params'] = trans_params
        return coeffs
    
    def work(self):
        for f in tqdm(self.datalist):
            dir, basename = f.split(os.sep)[-2], f.split(os.sep)[-1]
            basename = os.path.splitext(basename)[0]
            coeffs = self.prepare(f)
            if coeffs is None:
                continue
            coeffs_file = os.path.join(self.coeff_root, dir+'-'+basename+'.mat')
            new_coeffs = {}
            for k,v in coeffs.items():
                new_coeffs[k] = v.cpu().numpy()
            savemat(coeffs_file, new_coeffs)       


if __name__=='__main__':
    data_list='/home/songyifei9/Dataprepare/FacialAttribute/AIML-Human-Attributes-Detection-with-Facial-Feature-Extraction-master/results/dirty_images.txt'
    dataset = ImgDataset(data_list=data_list)
    dataset.work()



