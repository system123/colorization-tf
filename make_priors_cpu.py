import numpy as np
import os
from skimage.io import imread
from skimage import color
import multiprocessing as mp
from scipy.spatial.distance import cdist
from tqdm import tqdm
from optparse import OptionParser
import sys

parser = OptionParser()
parser.add_option("-f", "--file", dest="file_path", help="Path for the txt file containing training images")
(options, args) = parser.parse_args()

points = np.load('resources/pts_in_hull.npy')
points = points.astype(np.float64)
filename_lists = []
probs = np.zeros((313), dtype=np.float64)
num = 0

with open(options.file_path) as lists_f:
	for img_f in lists_f:
		filename_lists.append( img_f.strip() )

def calculate_img_priors(job):
	img_f, bins, queue = job
	# Try load the image file
	try:
		img = imread(img_f)
	except:
		print('Failed to load ' + img_f)
		sys.stdout.flush()
		return

	# Make sure the image is loaded correctly and they are RGB images
	if img is None or len(img.shape) != 3 or img.shape[2] != 3:
		return
	
	img_lab = color.rgb2lab(img)
	# Flatten each image layer
	img_lab = img_lab.reshape((-1, 3))
	img_ab = img_lab[:, 1:]

	H = cdist(img_ab, points, 'sqeuclidean').argmin(axis=1)
	stats = np.unique(H, return_counts=True)

	queue.put(stats)

def closest_idx(pt, bins):
	return(cdist([pt], bins, 'sqeuclidean').argmin())

def process_priors(queue):
	priors = np.zeros((313))

	while 1:
		stats = queue.get()

		if m == 'reduce':
			print("Reducing priors")
			priors /= priors.sum()
			print("Done Calculating Priors")
			np.save("prior_probs", priors)
			break
		else:
			priors[stats[0]] += stats[1]

manager = mp.Manager()
queue = manager.Queue()
pool = mp.Pool(10)

reducer = pool.apply_async(process_priors, (queue,))

jobs = [(img_f, points, queue) for img_f in filename_lists]

for _ in tqdm(pool.imap_unordered(calculate_img_priors, jobs), total=len(jobs)):
	pass

queue.put("reduce")
pool.close()
pool.join()

