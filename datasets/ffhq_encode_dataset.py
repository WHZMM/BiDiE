# UTF-8
# refer to datasets/images_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from glob import glob
import torchvision.transforms as transforms


class TrainDataset(Dataset):
	def __init__(self):
		self.mask_paths = sorted(glob("/home/hzwu/data_using/FFHQ-EG3D/FFHQ-EG3D-align_mask/?????.png"))
		self.photo_paths = sorted(glob("/home/hzwu/data_using/FFHQ-EG3D/FFHQ-EG3D/?????.jpg"))
		self.camera = np.load("/home/hzwu/dataset/FFHQ/camera_matrix.npy")  # (70000, 25)
		self.camera_mirror = np.load("/home/hzwu/dataset/FFHQ/camera_matrix_mirror.npy")  # (70000, 25)
		assert len(self.photo_paths) == len(self.mask_paths) == self.camera.shape[0] == self.camera_mirror.shape[0]
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
		if index < 70000:
			photo = Image.open(self.photo_paths[index]).convert("RGB")
			mask = Image.open(self.mask_paths[index]).convert("L")
			camera = self.camera[index]
		else:
			index -= 70000
			photo = Image.open(self.photo_paths[index]).convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
			mask = Image.open(self.mask_paths[index]).convert("L").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
			camera = self.camera_mirror[index]

		photo = self.photo_transform(photo)  # torch.Size([3, 512, 512])
		mask = self.mask_transform(mask)  # torch.Size([1, 512, 512])
		camera = torch.FloatTensor(camera)  # torch.Size([25])

		return photo, photo, mask, camera


class ValDataset(Dataset):
	def __init__(self):
		print("CelebA-HQ encode...")
		self.mask_paths = sorted(glob("/home/hzwu/data_using/CelabA-HQ-EG3D/CelebA-HQ-EG3D-align_mask/??????.png"))
		self.photo_paths = sorted(glob("/home/hzwu/data_using/CelabA-HQ-EG3D/CelebA-HQ-EG3D/??????.jpg"))
		self.camera = np.load("/home/hzwu/data_using/CelabA-HQ-EG3D/camera_matrix.npy")  # (29???, 25)
		self.camera_mirror = np.load("/home/hzwu/data_using/CelabA-HQ-EG3D/camera_matrix_mirror.npy")  # (29???, 25)
		assert len(self.photo_paths) == len(self.mask_paths) == self.camera.shape[0] == self.camera_mirror.shape[0]
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
		if index < 29983:
			photo = Image.open(self.photo_paths[index]).convert("RGB")
			mask = Image.open(self.mask_paths[index]).convert("L")
			camera = self.camera[index]
		else:
			index -= 29983
			photo = Image.open(self.photo_paths[index]).convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
			mask = Image.open(self.mask_paths[index]).convert("L").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
			camera = self.camera_mirror[index]

		photo = self.photo_transform(photo)  # torch.Size([3, 512, 512])
		mask = self.mask_transform(mask)  # torch.Size([1, 512, 512])
		camera = torch.FloatTensor(camera)  # torch.Size([25])

		return photo, photo, mask, camera


if __name__ == "__main__":
	print("print this only when test code : datasets/ffhq_encode_dataset.py")
	train_data = TrainDataset()
	print(len(train_data))
	val_data = ValDataset()
	print(len(val_data))

	a, b, c, d = train_data[0]
	print(d)
	a, b, c, d = train_data[70000]
	print(d)
	a, b, c, d = val_data[0]
	print(d)
	a, b, c, d = val_data[29983]
	print(d)

	for i in range(0, 1):
		a, b, c, d = train_data[i]
		print(b.shape, c.shape, d.shape)
		# # 
		# a = a.numpy()
		# a = (a + 1) * 127.5
		# a = np.around(a)
		# a = np.clip(a, 0.0, 255.0)
		# a = a.astype(np.uint8)
		# a = np.squeeze(a)
		# img = Image.fromarray(a).convert("L")
		# # img.save("/home/hzwu/data_using/temp/{:0>6}.png".format(i))
		print(c)
