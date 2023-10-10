import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from model.py import *


device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

total_dataset_path="./data/processed/total"
train_dataset_path="./data/processed/train"
val_dataset_path="./data/processed/val"

train_dataset_inmemory=AnimalFaces_Dataset_InMemory(train_dataset_path).to(device)
val_dataset_inmemory=AnimalFaces_Dataset_InMemory(val_dataset_path).to(device)

train_dataloader_vae_inmemory = DataLoader(train_dataset_inmemory,batch_size=512, shuffle=True)
val_dataloader_vae_inmemory = DataLoader(val_dataset_inmemory,batch_size=512, shuffle=True)

enc_layers=[(49152,10000),(10000,1000),(1000,50)]
dec_layers=[(25,1000),(1000,10000),(10000,49152)]

save_path="./model_weights"

vae_model_m2=VAE_m2(enc_layers,dec_layers).to(device)
#vae_model_m2.load_state_dict(torch.load("/home/chandan/ADRL_Assignment/model_weights/vae_model_m2_1_0d1.pth"))
optzr_m2=torch.optim.Adam(vae_model_m2.parameters(),lr=0.001, weight_decay=0.0001)

training_outputs_m2 = train_vae(vae_model_m2,optzr_m2,VAE_Loss,train_dataloader_vae_inmemory,val_dataloader_vae_inmemory,100,device,"vae_model_m2_L1_B1",save_path,L=1,beta=1.0)
vae_model_m2.eval()

num_samples = 10
gen_img=vae_model_m2.generate(num_samples=num_samples)

for i in range(num_samples):
    img=transform_toimage(gen_img.view(num_samples,3,128,128)[i])
    plt.imshow(img)
    plt.axis('off')  
    plt.show()