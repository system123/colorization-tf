from glob import glob
import multiprocessing as mp
import os
from PIL import Image, ImageCms
import numpy as np
from tqdm import tqdm
from skimage import color, io

def rgb2lab(img,reverse=False):
    if img.mode != "RGB":
        img = img.convert("RGB")

    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile  = ImageCms.createProfile("LAB")

    if reverse:
        transform = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")
    else:
        transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    
    out_img = ImageCms.applyTransform(img, transform)
    
    return(out_img)


def process_img(img_path):
    fname = os.path.basename(img_path)
    fout = os.path.join(out_dir,fname)
    if not os.path.exists(fout):
        img_sar = Image.open(img_path)
        img_opt = Image.open(os.path.join(opt_dir,fname))
        try:
            img_sar.verify()
            img_opt.verify()
        except Exception:
            print("Error with image: {}".format(img_path))
        else:
            #img_sar = Image.open(img_path)
            #img_opt = Image.open(os.path.join(opt_dir,fname))
            img_sar = io.imread(img_path)
            img_opt = io.imread(os.path.join(opt_dir,fname))
            
            #img_lab = rgb2lab(img_opt)
            #img_opt = np.asarray(img_opt, dtype=np.float32)
            img_lab = color.rgb2lab(img_opt)
            #img_lab = np.asarray(img_lab, dtype=np.float32)
            #img_sar = np.asarray(img_sar, dtype=np.float32)
            #print(img_lab[:,:,0].max())
            #print(img_lab[:,:,1].max())
            #print(img_sar.max())
            img_lab[:,:,0] = img_sar / 255.0 * 100

            #fused_img = Image.fromarray(img_lab.astype(np.uint8))
            #fused_img = rgb2lab(fused_img,reverse=True)

            #fused_img.save(fout)
            img_fus = color.lab2rgb(img_lab)
            io.imsave(fout,img_fus)
    return(0)


sar_dir = "/data/ne63wog/ihsPrediction/sar/"
opt_dir = "/data/ne63wog/ihsPrediction/opt/"

out_dir = "/data/ne63wog/ihsPrediction/lab/"

os.makedirs(out_dir, exist_ok=True)

pool = mp.Pool(4)

imgs = glob(os.path.join(sar_dir, "*.png"))

for _ in tqdm(pool.imap_unordered(process_img, imgs), total=len(imgs)):
    pass

pool.close()
pool.join()

