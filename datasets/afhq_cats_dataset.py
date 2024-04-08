# UTF-8

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
from glob import glob
import torchvision.transforms as transforms


class TrainDataset(Dataset):
	def __init__(self):
		self.photo_paths = sorted(glob("/home/hzwu/data_using/AFHQ_v2_cats/train/cat/*.png"))
		with open("/home/hzwu/data_using/AFHQ_v2_cats/dataset.json", "r") as fp:
			self.camera = json.load(fp)
			self.camera = self.camera["labels"]  # list of lists, every one of which is ["xxx.png", [x, x, x, ... , x]]
		self.camera = {k: v for k, v in self.camera}  # dict, key is file name, value is a list of 25 nums
		self.photo_transform = transforms.Compose([
			transforms.Resize((256, 256)), 
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		self.mask_transform = transforms.Compose([
			transforms.Resize((256, 256)), 
			transforms.ToTensor()])
		

	def __len__(self):  # mirror
		return 2 * len(self.photo_paths)

	def __getitem__(self, index):
		if index < 5065:
			photo = Image.open(self.photo_paths[index]).convert("RGB")
			camera = self.camera[self.photo_paths[index].rsplit("/", maxsplit=1)[-1]]
		else:
			index -= 5065
			photo = Image.open(self.photo_paths[index]).convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
			camera = self.camera[self.photo_paths[index].rsplit("/", maxsplit=1)[-1][:-4] + "_mirror.png"]

		photo = self.photo_transform(photo)
		camera = torch.FloatTensor(camera) 

		return photo, camera

if __name__ == "__main__":
	print("print this line only when test code : datasets/ffhq_encode_dataset.py")
	train_data = TrainDataset()

	print(len(train_data))
	for i in range(len(train_data)):
		a, b = train_data[i]
		print(a.shape, b.shape)
		print(".", end="")
		break
