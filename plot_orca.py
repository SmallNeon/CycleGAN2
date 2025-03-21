# This code is released under the CC BY-SA 4.0 license.
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Check if the environment variable is set
if os.getenv('KMP_DUPLICATE_LIB_OK') == 'TRUE':
    print("Environment variable set successfully.")
else:
    print("Failed to set environment variable.")


import os
import SimpleITK as sitk
import numpy as np
import torch
# import pickle

from models import create_model
from options.train_options import TrainOptions
import matplotlib.pyplot as plt


# def load_array(file_path):
#     with open(file_path, 'rb') as f:
#         loaded_data = pickle.load(f)
#         return loaded_data

@torch.no_grad()
def save_fake_native(out_path,tagA='ARTERIAL', tagB='NATIVE', device='cpu',data_path = ''):
    # root_path - is the path to the raw Coltea-Lung-CT-100W data set.

    opt = TrainOptions().parse()
    opt.load_iter = 90
    opt.isTrain = False
    opt.device = device
    opt.model = 'da_cytran'
    opt.name = 'da_all'

    model = create_model(opt)
    model.setup(opt)
    gen = model.netG_A
    # gen = model.netG_B
    gen.eval()

    # eval_dirs = pd.read_csv(os.path.join(root_path, 'test_data.csv'))
    # eval_dirs = list(eval_dirs.iloc[:, 1])
    # eval_dirs = ['tev3p2',"tev3p4","tev3p5","tev3p6","tev3p7"]
    # test_dirs = ['tev2p2','trv2p2','trv2p3','trv2p8','vav4p1']
    # eval_dirs = [i for i in os.listdir(r'C:\Users\qzhuang4\Desktop\Orca2d\v1') if i not in test_dirs]
    # eval_dirs = [i for i in os.listdir(r'C:\Users\qzhuang4\Desktop\56Orca2d')][:20]
    eval_dirs = [i for i in os.listdir(data_path)]
    # test_data = load_array(r'C:\Users\qzhuang4\Desktop\cycle-transformer\datasets\testAll.pickle')
    # eval_dirs = [i.split('\\')[-3] for i in test_data if 'patient' in i]
    # print(eval_dirs[0])

    for patient in os.listdir(data_path):

        if not patient in eval_dirs:
            continue

        patient_dir = os.path.join(data_path,patient)
        for scan in os.listdir(os.path.join(patient_dir,tagA)): 
            orig_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_dir,tagA,scan)))
            # patient, sclice = scan.split('\\')[-3], scan.split('\\')[-1]
            new_folder_path = os.path.join(out_path,patient)    
            
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)   

            # Scale original image, which is transform
            # orig_img[orig_img < 0] = 0
            orig_img = orig_img / 1e3
            # orig_img = orig_img - 1

            orig_img_in = np.expand_dims(orig_img, 0).astype(np.float64)
            orig_img_in = torch.from_numpy(orig_img_in).float().to(device)
            orig_img_in = orig_img_in.unsqueeze(0)

            native_fake = gen(orig_img_in)[0, 0].detach().cpu().numpy()

            img_corr = sitk.GetImageFromArray(native_fake)
            sitk.WriteImage(img_corr, os.path.join(new_folder_path,scan))


def plot_fake_image(patient,slice):
    path = os.path.join(r'C:\Users\qzhuang4\Desktop\cycle-transformer\predictions-orca56',patient,slice)
    img = sitk.ReadImage(path)
    img_npy = sitk.GetArrayFromImage(img) 
    plt.imshow(img_npy, cmap='gray')
    plt.show()

            
if __name__ == '__main__':
    save_fake_native(
        out_path = r'C:\Users\qzhuang4\Desktop\cycle-transformer\predictions-da-all\uci',
        device='cuda',
        data_path = r'C:\Users\qzhuang4\Desktop\uci2d'
    )
