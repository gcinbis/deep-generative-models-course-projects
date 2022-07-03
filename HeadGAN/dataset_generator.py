import sys
import argparse
import textwrap
import cv2
import yaml
import numpy as np
from PIL import Image
import os
import datetime
import torch
import torchvision 

sys.path += [".", "./TDDFA_V2/"]

from TDDFA_V2.FaceBoxes import FaceBoxes
from TDDFA_V2.TDDFA import TDDFA
from TDDFA_V2.utils.render import render
#from TDDFA_V2.utils.render_ctypes import render  # faster
from TDDFA_V2.utils.depth import depth
from TDDFA_V2.utils.pncc import pncc
from TDDFA_V2.utils.uv import uv_tex
from TDDFA_V2.utils.pose import viz_pose
from TDDFA_V2.utils.serialization import ser_to_ply, ser_to_obj
from TDDFA_V2.utils.functions import draw_landmarks, get_suffix
from TDDFA_V2.utils.tddfa_util import str2bool
import sys
import os

initial_working_dir = os.getcwd()
os.chdir(initial_working_dir + "/TDDFA_V2/")

config = 'configs/mb1_120x120.yml'
cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
opt = 'pncc'
show_flag = False

os.chdir(initial_working_dir)

def get_xt(img, boxes, param_lst, roi_box_lst, img2, opt, show_flag, tddfa, face_boxes):
	# Get the geometry images of the driving face images with static face characteristics
	
	boxes2 = face_boxes(img2)
	
	n = len(boxes2)
	
	if n == 0:
		return None
	
	param_lst2, roi_box_lst2 = tddfa(img2, boxes2)

	# Take face characteristics from the static face and
	# orientation and mimmics from the driving face.
	new_params = np.concatenate((param_lst2[0][0:12], param_lst[0][12: 52], param_lst[0][52:62])).reshape(1, 62)

	wfp = None
	
	dense_flag = opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
	ver_lst = tddfa.recon_vers(new_params, roi_box_lst, dense_flag=dense_flag)

	pncc_result = pncc(img, ver_lst, tddfa.tri, show_flag=show_flag, wfp=wfp, with_bg_flag=False)
	
	return pncc_result

def get_yt(img, boxes, param_lst, roi_box_lst, opt, show_flag, tddfa):
	# Get the geometry image of the static face

	wfp = None
	
	dense_flag = opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
	ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
	pncc_result = pncc(img, ver_lst, tddfa.tri, show_flag=show_flag, wfp=wfp, with_bg_flag=False)
	
	return pncc_result


def process_pair(img, img2s, face_boxes, tddfa):
	# Get both static and driving face visual features

	# choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
	
	boxes = face_boxes(img)
	n = len(boxes)
	
	if n == 0:
		return None, None
	
	param_lst, roi_box_lst = tddfa(img, boxes)
	
	yt = get_yt(img, boxes, param_lst, roi_box_lst, opt, show_flag, tddfa)
	xts = []	
	for img2 in img2s:
		if img2 is None:
			return None, None
		new_xt = get_xt(img, boxes, param_lst, roi_box_lst, img2, opt, show_flag, tddfa, face_boxes)
		if img2 is None:
			return None, None
		xts.append(new_xt)
	return yt, xts

# https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black
def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
	x, y = im.size
	size = max(min_size, x, y)
	new_im = Image.new('RGB', (size, size), fill_color)
	new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
	return new_im.resize((256, 256))

def get_data_batch(paths, frame_per_path, is_gpu, debug_output=False):
	# Given an array of video paths, generate a batch of images out of these videos 
	# (currently ignoring the audio features but providing Gaussian random vectors instead)
	image_array = []
	image_pncc_array = []
	transfer_pncc_array = []
	audio_array = []
	ground_truth_image_array = []

	norm_image_array = []
	norm_image_pncc_array = []
	norm_transfer_pncc_array = []
	norm_audio_array = []
	norm_ground_truth_image_array = []

	# The ONNX version seems to be faster but it is good to keep 
	# the other option to not forget about it in the future.
	if True:
		import os
		os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
		os.environ['OMP_NUM_THREADS'] = '16'
		from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
		from TDDFA_ONNX import TDDFA_ONNX
		os.chdir(initial_working_dir + "/TDDFA_V2/")
		face_boxes = FaceBoxes_ONNX()
		tddfa = TDDFA_ONNX(gpu_mode=is_gpu, **cfg)
		os.chdir(initial_working_dir)
	else:
		tddfa = TDDFA(gpu_mode=is_gpu, **cfg)
		face_boxes = FaceBoxes()

	v_index = 0
	for video_path in paths:
		videocap = cv2.VideoCapture(video_path)
		success,image = videocap.read()
		success = True
		count = 0

		# time stamp and image matching was for audio features but cancelled later due to 
		# the reasons mentioned in the main.ipynb
		images = []
		timestamps = []
		while success:
			success,image = videocap.read()

			if success:
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

				time_stamp = float(videocap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
				if time_stamp == 0.0:
					break
				image = Image.fromarray(image).convert("RGB")
				image = make_square(image)

				images.append(np.asarray(image))
				timestamps.append(time_stamp)
				transcription_index = 0
				
		for i in range(frame_per_path):
			# A hard limit to prevent out of bounds access
			if len(images) < 8:
				continue
			i_static = np.random.randint(4, len(images) - 3) # L for sound data is 4
			i_dynamic = np.random.randint(4, len(images) - 3) # L for sound data is 4
			im_static = images[i_static]
			im_dynamics = [images[i_dynamic], images[i_dynamic - 1], images[i_dynamic - 2]]
			yt, xts = process_pair(im_static, im_dynamics, face_boxes, tddfa)

			# Normalization of the image data
			if yt is not None and xts is not None: 
				transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize(
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225],
				),
				torchvision.transforms.ToPILImage(),
				])
				nomalized_im_static = np.asarray(transform(im_static))
				normalized_ground_truth_image = np.asarray(transform(im_dynamics[0]))

				norm_image_array.append(nomalized_im_static)
				norm_ground_truth_image_array.append(normalized_ground_truth_image)
				norm_image_pncc_array.append(yt)
				norm_transfer_pncc_array.append(xts)
				norm_audio_array.append(np.random.normal(0.0, 1.0, 300))

				image_array.append(im_static)
				ground_truth_image_array.append(im_dynamics[0])
				image_pncc_array.append(yt)
				transfer_pncc_array.append(xts)
				audio_array.append(np.random.normal(0.0, 1.0, 300))

				if debug_output:	
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "im_static.png", im_static)
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "im_dynamic.png", im_dynamics[0])
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "yt.png", yt)
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "xt0.png", xts[0])
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "xt1.png", xts[1])
					cv2.imwrite("test_outputs/" + str(v_index) + "_" + str(i) + "_" + "xt2.png", xts[2])

			v_index += 1
	if debug_output:
		print(len(image_pncc_array))
		print(len(image_array))
		print(len(image_pncc_array))
		print(len(transfer_pncc_array))
		print(len(audio_array))

	return norm_image_array, norm_image_pncc_array, norm_transfer_pncc_array, norm_audio_array, ground_truth_image_array, (image_array, image_pncc_array, transfer_pncc_array, audio_array, ground_truth_image_array)

if __name__ == '__main__':
	image_array, image_pncc_array, transfer_pncc_array, audio_array, ground_truth_image_array, unnormalized = get_data_batch(['TrainingData/vox2_mp4/mp4/id00017/01dfn2spqyE/00001.mp4'], 1, is_gpu=True, debug_output=True)

# image_array: Static face image (1024 np arrays for example)
# image_pncc_array: Geometry image of the static face image (pncc) (1024 np arrays for example)
# transfer_pncc_array: Geometry images of the driving faces (pncc) (1024 lists, each containing three np arrays)
# audio_array: audio features, (1024 np arrays): Not real audio features but Gaussian random vectors instead
# ground_truth_image_array: Ground truth image of the driving face (1024 tane np array)

