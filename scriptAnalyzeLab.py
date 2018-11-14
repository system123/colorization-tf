#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:35:45 2018

@author: michael
"""
from optparse import OptionParser
from skimage.io import imread
from skimage import color
import numpy as np
import sys
import multiprocessing as mp
from tqdm import tqdm

parser = OptionParser()
parser.add_option("-f", "--file", dest="file_path",
                  help="Path for the txt file containing training images")
(options, args) = parser.parse_args()

def process_img(job):
    img_f, queue = job

    a_range = np.linspace(-95.0,105.0,21)
    b_range = np.linspace(-115.0,105.0,23)

    img_f = img_f.strip()
    try: 
        img = imread(img_f)    
    except:
        print('Failed to load '+img_f)
        sys.stdout.flush()
        return

    if img is None or len(img.shape)!=3 or img.shape[2]!=3:
        print('Problem occured!')
        sys.stdout.flush()
        return

    img_lab = color.rgb2lab(img)
    img_lab = img_lab.reshape(-1,3)
    img_ab = img_lab[:, 1:]
    hist_ab = np.histogram2d(img_ab[:,0],img_ab[:,1], bins=(a_range, b_range))    
    hist_ab = hist_ab[0]

    queue.put(hist_ab)

def reduce_queue(queue):
    a_range = np.linspace(-95.0,105.0,21)
    b_range = np.linspace(-115.0,105.0,23)
    a_centers = (a_range[:-1] + a_range[1:])/2
    b_centers = (b_range[:-1] + b_range[1:])/2

    points = np.load('resources/pts_in_hull.npy')
    points = points.astype(np.float64)
    prior_probs = np.zeros((313), dtype=np.float64)

    sum_hist_ab = np.zeros((a_centers.shape[0],b_centers.shape[0]))

    while True:
        hist_ab = queue.get()
               
        if isinstance(hist_ab, str) and hist_ab == "done":
            print("Reducing prior probs")        
            sum_hist_ab += 1e-6
            completeCount = np.sum(sum_hist_ab)
            #print(completeCount.astype(np.float64))
            probs = (1.0/completeCount) * sum_hist_ab

            for i in tqdm(range(0,a_centers.shape[0])):
                for j in range(0,b_centers.shape[0]):
                    ind = np.where(np.all((a_centers[i],b_centers[j])==points,axis=1))
                    ind = ind[0].astype(int)

                    if ind.size>0:
                        prior_probs[ind] = probs[i,j]
                        
            np.save('resources/prior_probs_michael', prior_probs)
            break
        else:
            sum_hist_ab = sum_hist_ab + hist_ab

filename_lists = []
with open(options.file_path) as lists_f:
    for img_f in lists_f:
        filename_lists.append( img_f.strip() )

manager = mp.Manager()
queue = manager.Queue()
pool = mp.Pool(10)

reducer = pool.apply_async(reduce_queue, (queue,))

jobs = [(img_f, queue) for img_f in filename_lists]

for _ in tqdm(pool.imap_unordered(process_img, jobs), total=len(jobs)):
    pass

queue.put("done")
pool.close()
pool.join()
