import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import os
import numpy as np
import time
import matplotlib.pyplot as plt

transform_totensor = transforms.Compose([
    transforms.ToTensor(),
])

transform_toimage = transforms.ToPILImage()

class AnimalFaces_Dataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.file_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        file_path = os.path.join(self.folder_path, file_name)
        img=transform_totensor(Image.open(file_path))
        return img.view(-1)
    

class AnimalFaces_Dataset_InMemory(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        file_list = os.listdir(folder_path)
        self.data={}
        count=0
        for file_name in file_list:
            count+=1
            file_path = os.path.join(folder_path, file_name)
            img=transform_totensor(Image.open(file_path))
            self.data[count]=img.view(-1).tolist()
        self.data=torch.tensor(list(self.data.values()))

    def to(this,device):
        this.data=this.data.to(device)
        return this
        
    def __len__(self):
        return self.data.shape[0]
    
    def shape(this):
        return this.data.shape

    def __getitem__(self, index):
        return self.data[index]


class MLP(nn.Module):
    def __init__(this,layers_lst):
        super().__init__()
        this.layers=nn.ModuleList()
        for in_dim, out_dim in layers_lst:
            this.layers.append(nn.Linear(in_dim,out_dim))
        this.activation=nn.Sigmoid()
    
    def forward(this,x):
        x=x
        for lyr in this.layers[:-1]:
            x=lyr(x)
            x=this.activation(x)
        return this.layers[-1](x)
    

class VAE_m2(nn.Module):
    def __init__(this,enc_layers,dec_layers):
        if(enc_layers[-1][1] != 2*dec_layers[0][0]):
            print("ERROR:Latent variable dimension missmatch with encoder output dimension")
            return
        super().__init__()
        this.ENC=MLP(enc_layers)
        this.DEC=nn.Sequential(MLP(dec_layers),nn.Sigmoid())
        this.z_dim=dec_layers[0][0]
        this.x_dim=dec_layers[-1][1]
        this.MVG=None
    
    def forward(this,x,L=1):
        device=next(this.parameters()).device
        if(this.MVG==None):
            mean=torch.zeros(this.z_dim).to(device)
            var=torch.eye(this.z_dim).to(device)
            this.MVG=torch.distributions.MultivariateNormal(mean, var)

        #Encoder
        x=this.ENC(x)

        #sampling
        mean_batch=x[:,:this.z_dim]
        sd_batch=x[:,this.z_dim:]

        z=torch.empty(0,this.z_dim,device=device)
        for _ in range(L):
            z=torch.cat((z,this.MVG.sample().unsqueeze(dim=0)),dim=0)
        
        #z=torch.matmul(z,var_clk_batch.view(var_clk_batch.shape[0],this.z_dim,this.z_dim))
        temp_z=torch.empty(0,this.z_dim,device=device)

        for mean_idx in range(mean_batch.shape[0]):
            temp_z=torch.cat((temp_z,(z*sd_batch[mean_idx])+mean_batch[mean_idx]),dim=0)

        #Decoder
        x=this.DEC(temp_z)

        return x , mean_batch, sd_batch
    
    def generate(this,x=None,num_samples=1):
        device=next(this.parameters()).device
        if(x!=None):
            x_mean, z_mean_batch, z_sd_batch = this.forward(x)
            return x_mean
        
        else:
            if(this.MVG==None):
                mean=torch.zeros(this.z_dim).to(device)
                var=torch.eye(this.z_dim).to(device)
                this.MVG=torch.distributions.MultivariateNormal(mean, var)
            
            z=torch.empty(0,this.z_dim,device=device)
            for _ in range(num_samples):
                z=torch.cat((z,this.MVG.sample().unsqueeze(dim=0)),dim=0)
            
            x_mean=this.DEC(z)

            return x_mean
    
    def postirior_infer(this,x,L=1):
        device=next(this.parameters()).device
        if(this.MVG==None):
            mean=torch.zeros(this.z_dim).to(device)
            var=torch.eye(this.z_dim).to(device)
            this.MVG=torch.distributions.MultivariateNormal(mean, var)

        #Encoder
        x=this.ENC(x)

        #sampling
        mean_batch=x[:,:this.z_dim]
        sd_batch=x[:,this.z_dim:]

        z=torch.empty(0,this.z_dim,device=device)
        for _ in range(L):
            z=torch.cat((z,this.MVG.sample().unsqueeze(dim=0)),dim=0)
        
        #z=torch.matmul(z,var_clk_batch.view(var_clk_batch.shape[0],this.z_dim,this.z_dim))
        temp_z=torch.empty(0,this.z_dim,device=device)

        for mean_idx in range(mean_batch.shape[0]):
            temp_z=torch.cat((temp_z,(z*sd_batch[mean_idx])+mean_batch[mean_idx]),dim=0)

        return mean_batch, sd_batch, temp_z

def VAE_Loss(x,output,beta=1.0):
    x_pred=output[0]
    mean_batch=output[1]
    sd_batch=output[2]
    L=x_pred.shape[0]//mean_batch.shape[0]
    loss=0.0
    Rec_loss=0.0
    KL_loss=0.0

    for x_idx in range(x.shape[0]):
        s_idx=x_idx*L
        loss1=(((x_pred[s_idx: s_idx+L,:] - x[x_idx])**2).sum() / L)
        loss2=(beta*((mean_batch[x_idx]**2).sum()+(sd_batch[x_idx]**2).sum()-(torch.log(sd_batch[x_idx]**2)).sum()-1))
        loss += loss1+loss2
        Rec_loss+=loss1
        KL_loss+=loss2
    
    return loss , Rec_loss, KL_loss


def train_vae(model,optimizer,loss_func,train_dataloader,val_dataloader,num_epoch,device,model_name,save_path=None,L=1,beta=1.0):
    def save_curve(lst,file_name):
        temp=''
        for i in lst:
            temp+=str(i)+'  '
        with open(save_path+"/"+file_name,'w') as f:
            f.write(temp)

    model.train()
    train_loss_curve=[]
    train_rec_loss_curve=[]
    train_kl_loss_curve=[]
    val_rec_loss_curve=[]
    val_kl_loss_curve=[]
    val_loss_curve=[]
    st=time.time()
    last_val_loss=float('inf')
    print("________________ Training VAE ________________\n")
    for epoch in range(num_epoch):
        print("__________________________")
        print("Epoch:",epoch)
        est=time.time()
        train_loss=0
        train_rec_loss=0
        train_kl_loss=0
        val_loss=0
        val_rec_loss=0
        val_kl_loss=0
        batch_count=0
        for batch in train_dataloader:
            temp_st=time.time()
            batch_count+=1
            optimizer.zero_grad()
            batch=batch.to(device)
            output=model(batch,L)
            loss, rec_loss, kl_loss = loss_func(batch,output,beta)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_rec_loss+=rec_loss.item()
            train_kl_loss+=kl_loss.item()
            temp_et=time.time()
            #print("training batch_count:",batch_count," time taken:",temp_et-temp_st," sec")

        
        for batch in val_dataloader:
            #optimizer.zero_grad()
            batch=batch.to(device)
            output=model(batch,L)
            loss, rec_loss, kl_loss=loss_func(batch,output,beta)
            val_loss+=loss.item()
            val_rec_loss+=rec_loss.item()
            val_kl_loss+=kl_loss.item()
        train_loss_curve.append(train_loss)
        train_rec_loss_curve.append(train_rec_loss)
        train_kl_loss_curve.append(train_kl_loss)
        val_loss_curve.append(val_loss)
        val_rec_loss_curve.append(val_rec_loss)
        val_kl_loss_curve.append(val_kl_loss)
        eet=time.time()
        print("training loss:",train_loss)
        print("val loss:",val_loss)
        print("time taken:",eet-est," sec")
        if(save_path!=None and last_val_loss>=val_loss):
            last_val_loss=val_loss
            torch.save(model.state_dict(), save_path+"/"+model_name+".pth")
            print("---weights saved---")
        print("__________________________\n")

    #saving loss curves
    save_curve(train_loss_curve,model_name+"_train_loss.txt")
    save_curve(train_rec_loss_curve,model_name+"_train_rec_loss.txt")
    save_curve(train_kl_loss_curve,model_name+"_train_kl_loss.txt")
    save_curve(val_loss_curve,model_name+"_val_loss.txt")
    save_curve(val_rec_loss_curve,model_name+"_val_rec_loss.txt")
    save_curve(val_kl_loss_curve,model_name+"_val_kl_loss.txt")
    
    et=time.time()
    print("Training done")
    print("Total time taken:",et-st," sec")
    print("______________________________________________")

    return model, train_loss_curve, train_rec_loss_curve, train_kl_loss_curve, val_loss_curve, val_rec_loss_curve, val_kl_loss_curve
    

        

