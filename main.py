
import torch
import argparse
import os
import random
from glob import glob
from torch import nn
import copy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import importlib
from sklearn.metrics import mean_squared_error

import models
import compare

importlib.reload(models)
importlib.reload(compare)


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

def adjust_learning_rate(optimizer, epoch, learing_rate):
    lr = learing_rate*pow(0.7,int(epoch/50))
    #print("lr=",lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','FashionMNIST','cifar10','HAM','CelebA'])
parser.add_argument('--n_epochs', type=int, default=302, help='number of epochs of training')
parser.add_argument('--floder', type=str, default='images', help=' ')
parser.add_argument('--N', type=int, default=2, help='number of epochs of training')
parser.add_argument('--lr1', type=float, default=0.001, help='number of epochs of training')
parser.add_argument('--lr2', type=float, default=0.001, help='number of epochs of training') #0.02
parser.add_argument('--split', type=int, default=1, help='number of epochs of training')

#args = parser.parse_args(args=[])
args = parser.parse_args()
print(args)


shape = 256
if args.split==1 or args.split==2:
    shape = 256
elif args.split==3:
    shape = 128
elif args.split==4:
    shape = 64




if args.dataset == 'MNIST':
    print('MNIST')
    tp=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_db = datasets.MNIST("/data0/HHong/data/mnist", train=True,download=True,transform=tp)
    test_db = datasets.MNIST("/data0/HHong/data/mnist", train=False, download=True,transform=tp)


    if args.split==1:
        client_modle = models.ResNet18_client_side1(models.Baseblock,[2,2,2],1,32).to(device)
        server_model = models.ResNet18_server_side1(models.Baseblock,[2,2,2], 10).to(device)
    elif args.split==2:
        client_modle = models.ResNet18_client_side2(models.Baseblock,[2,2,2],1,32).to(device)
        server_model = models.ResNet18_server_side2(models.Baseblock,[2,2,2], 10).to(device)
    elif args.split==3:
        client_modle = models.ResNet18_client_side3(models.Baseblock,[2,2,2],1,32).to(device)
        server_model = models.ResNet18_server_side3(models.Baseblock,[2,2,2], 10).to(device)
    elif args.split==4:
        client_modle = models.ResNet18_client_side4(models.Baseblock,[2,2,2],1,32).to(device)
        server_model = models.ResNet18_server_side4(models.Baseblock,[2,2,2], 10).to(device)
    


    client_modle_parameter = torch.load("/data0/HHong/split/mnist-split/mnist_cmd{}".format(args.split))
    client_modle.load_state_dict(client_modle_parameter)

    server_model_parameter = torch.load("/data0/HHong/split/mnist-split/mnist_smd{}".format(args.split))
    server_model.load_state_dict(server_model_parameter)


    gen = models.Generator32(1,100).to(device)
    res_gen = models.Generator32(1,64*shape).to(device)
    optimizer_gen = optim.Adam(gen.parameters(),lr=args.lr1)
    optimizer_res_gen = optim.Adam(res_gen.parameters(),lr=args.lr2)


if args.dataset == 'cifar10':
    print('cifar10')
    tp=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_db = datasets.CIFAR10("/data0/HHong/data/cifar10", train=True,download=True,transform=tp)
    test_db = datasets.CIFAR10("/data0/HHong/data/cifar10", train=False, download=True,transform=tp)


    # client_modle = models.ResNet18_client_side4(models.Baseblock,[2,2,2]).to(device)
    # server_model = models.ResNet18_server_side4(models.Baseblock,[2,2,2], 10).to(device)

    if args.split==1:
        client_modle = models.ResNet18_client_side1(models.Baseblock,[2,2,2],3,32).to(device)
        server_model = models.ResNet18_server_side1(models.Baseblock,[2,2,2], 10).to(device)
    elif args.split==2:
        client_modle = models.ResNet18_client_side2(models.Baseblock,[2,2,2],3,32).to(device)
        server_model = models.ResNet18_server_side2(models.Baseblock,[2,2,2], 10).to(device)
    elif args.split==3:
        client_modle = models.ResNet18_client_side3(models.Baseblock,[2,2,2],3,32).to(device)
        server_model = models.ResNet18_server_side3(models.Baseblock,[2,2,2], 10).to(device)
    elif args.split==4:
        client_modle = models.ResNet18_client_side4(models.Baseblock,[2,2,2],3,32).to(device)
        server_model = models.ResNet18_server_side4(models.Baseblock,[2,2,2], 10).to(device)



    client_modle_parameter = torch.load("/data0/HHong/split/cifar-split/cifar_cmd{}".format(args.split))
    client_modle.load_state_dict(client_modle_parameter)
    server_model_parameter = torch.load("/data0/HHong/split/cifar-split/cifar_smd{}".format(args.split))
    server_model.load_state_dict(server_model_parameter)



    gen = models.Generator32(3,100).to(device)
    res_gen = models.Generator32(3,64*shape).to(device)
    optimizer_gen = optim.Adam(gen.parameters(),lr=args.lr1)
    optimizer_res_gen = optim.Adam(res_gen.parameters(),lr=args.lr2)

if args.dataset == 'HAM':
    print('HAM')
    class SkinData(Dataset):
        def __init__(self, df, transform = None):

            self.df = df
            self.transform = transform

        def __len__(self):

            return len(self.df)

        def __getitem__(self, index):

            X = Image.open(self.df['path'][index]).resize((64, 64))
            y = torch.tensor(int(self.df['target'][index]))
            if self.transform:
                X = self.transform(X)
            return X, y

    def readData(path):
        df = pd.read_csv(path)
        lesion_type = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }
        imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join("/data0/HHong/data/HAM", '*', '*.jpg'))}
        df['path'] = df['image_id'].map(imageid_path.get)
        df['cell_type'] = df['dx'].map(lesion_type.get)
        df['target'] = pd.Categorical(df['cell_type']).codes
        # print(df['cell_type'].value_counts())
        # print(df['target'].value_counts())
        return  df
    df = readData('/data0/HHong/data/HAM/HAM10000_metadata.csv')
    train, test = train_test_split(df, test_size = 0.2)
    SFL_train, ResAtk = train_test_split(train, test_size = 0.5)
    ResAtk = ResAtk.reset_index()
    test = test.reset_index()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.Pad(3),
                            transforms.RandomRotation(10),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = mean, std = std)
                            ])
    test_transforms = transforms.Compose([
                            transforms.Pad(3),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = mean, std = std)
                            ])
    train_db = SkinData(ResAtk, transform = train_transforms)
    test_db = SkinData(test, transform = test_transforms)

    # client_modle = models.ResNet18_client_side4(models.Baseblock,[2,2,2]).to(device)
    # server_model = models.ResNet18_server_side4(models.Baseblock,[2,2,2], 7).to(device)
    if args.split==1:
        client_modle = models.ResNet18_client_side1(models.Baseblock,[2,2,2],3,64).to(device)
        server_model = models.ResNet18_server_side1(models.Baseblock,[2,2,2], 7).to(device)
    elif args.split==2:
        client_modle = models.ResNet18_client_side2(models.Baseblock,[2,2,2],3,64).to(device)
        server_model = models.ResNet18_server_side2(models.Baseblock,[2,2,2], 7).to(device)
    elif args.split==3:
        client_modle = models.ResNet18_client_side3(models.Baseblock,[2,2,2],3,64).to(device)
        server_model = models.ResNet18_server_side3(models.Baseblock,[2,2,2], 7).to(device)
    elif args.split==4:
        client_modle = models.ResNet18_client_side4(models.Baseblock,[2,2,2],3,64).to(device)
        server_model = models.ResNet18_server_side4(models.Baseblock,[2,2,2], 7).to(device)

    client_modle_parameter = torch.load("/data0/HHong/split/ham-split/ham_cmd{}".format(args.split))
    client_modle.load_state_dict(client_modle_parameter)
    server_model_parameter = torch.load("/data0/HHong/split/ham-split/ham_smd{}".format(args.split))
    server_model.load_state_dict(server_model_parameter)



    gen = models.Generator64(3,100).to(device)
    res_gen = models.Generator64(3,64*shape).to(device)
    optimizer_gen = optim.Adam(gen.parameters(),lr=args.lr1)
    optimizer_res_gen = optim.SGD(res_gen.parameters(),lr=args.lr2)

if args.dataset == 'CelebA':
    print('CelebA')
    tp=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.506, 0.425, 0.382), (0.265, 0.245, 0.241)),
    ])
    train_db = datasets.CelebA(root="/data0/HHong/data/celeba",split="train",transform=tp,download=True,target_type="attr")
    test_db = datasets.CelebA(root="/data0/HHong/data/celeba",split="test",transform=tp,download=True,target_type="attr")

    # client_modle = models.ResNet18_client_side1(models.Baseblock,[2,2,2]).to(device)
    # server_model = models.ResNet18_server_side1(models.Baseblock,[2,2,2], 2).to(device)

    if args.split==1:
        client_modle = models.ResNet18_client_side1(models.Baseblock,[2,2,2],3,64).to(device)
        server_model = models.ResNet18_server_side1(models.Baseblock,[2,2,2], 2).to(device)
    elif args.split==2:
        client_modle = models.ResNet18_client_side2(models.Baseblock,[2,2,2],3,64).to(device)
        server_model = models.ResNet18_server_side2(models.Baseblock,[2,2,2], 2).to(device)
    elif args.split==3:
        client_modle = models.ResNet18_client_side3(models.Baseblock,[2,2,2],3,64).to(device)
        server_model = models.ResNet18_server_side3(models.Baseblock,[2,2,2], 2).to(device)
    elif args.split==4:
        client_modle = models.ResNet18_client_side4(models.Baseblock,[2,2,2],3,64).to(device)
        server_model = models.ResNet18_server_side4(models.Baseblock,[2,2,2], 2).to(device)


    #client_modle = nn.DataParallel(client_modle).to(device)
    client_modle_parameter = torch.load("/data0/HHong/split/celeba-split/celeba_cmd{}".format(args.split))
    client_modle.load_state_dict(client_modle_parameter)
    #server_model = nn.DataParallel(server_model).to(device)
    server_model_parameter = torch.load("/data0/HHong/split/celeba-split/celeba_smd{}".format(args.split))
    server_model.load_state_dict(server_model_parameter)


    gen = models.Generator64(3,100).to(device)
    res_gen = models.Generator64(3,64*shape).to(device)
    optimizer_gen = optim.Adam(gen.parameters(),lr=args.lr1)
    optimizer_res_gen = optim.Adam(res_gen.parameters(),lr=args.lr2)




os.makedirs("./{}".format(args.floder), exist_ok=True)
os.makedirs("./{}".format(args.floder), exist_ok=True)
test_loader = DataLoader(test_db, batch_size=100, shuffle=False)
smashed_data = torch.tensor([1])
real_image = torch.tensor([1])
for real,label in test_loader:
    save_image(real[:64], "./{}/real.png".format(args.floder), nrow=8)
    real_image = real
    #print(real.size())
    smashed_data = client_modle(real.to(device))
    break
print(smashed_data.size())


for v in range(100):
    save_image(real_image[v], "./{}/rel_{}.png".format(args.floder,v), nrow=1)




####构建template####
data_loader = DataLoader(train_db, batch_size=500, shuffle=True)
smasheds = []
count = 0
for real,label in data_loader:
    smashed_ = client_modle(real.to(device)).detach()
    smasheds.append(smashed_)
    count+=1
    if count==args.N:
        break
template = smasheds[0]
for i in range(len(smasheds)):
    if i!=0:
        template = torch.cat([template,smasheds[i]],0)
print(template.size())


criterion = torch.nn.CrossEntropyLoss().to(device)
MSE = nn.MSELoss().to(device)


mse_score = 1000000
ssim_score = 0
PSNR_score = 0
LPIP_score = 1000000



for iter in range(args.n_epochs):
    gen.train()
    res_gen.train()

    data_loader2 = DataLoader(template, batch_size=128, shuffle=True)
    for bat_data in data_loader2:
        optimizer_gen.zero_grad()
        optimizer_res_gen.zero_grad()

        z = Variable(torch.randn(128, 100)).to(device)
        gen_imgs = gen(z)
        smash = client_modle(gen_imgs)
        outputs_T = server_model(smash)
        pred = outputs_T.data.max(1)[1]
        loss_one_hot = criterion(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss_gen = loss_one_hot  + loss_information_entropy*4  


        cycle_loss1 = MSE(gen_imgs, res_gen(smash))  

        bat_data = bat_data.to(device)
        cycle_loss3 = MSE(client_modle(res_gen(bat_data)),bat_data)   


        loss = cycle_loss1*1 +cycle_loss3*3 + loss_gen*0.1  #1 3 0.1
        loss.backward()
        
        optimizer_gen.step()
        optimizer_res_gen.step()


    #if iter%10 == 0:
    print("iter:{},oh:{},ie:{},c1:{},c2:{},c3:{},gen:{},res_gen:{}".format(iter,loss_one_hot.item(),loss_information_entropy.item(),cycle_loss1.item(),0,cycle_loss3.item(),loss_gen.item(),0))
    res_gen.eval()
    fake_im = res_gen(smashed_data)

    
    save_image(fake_im[:64], "./{}/res.png".format(args.floder), nrow=8)
    for v in range(100):
        save_image(fake_im[v], "./{}/res_{}.png".format(args.floder,v), nrow=1)


    #print("mse1:",MSE(fake_im,real_image.to(device)).item())

    mse = compare.calc_mse("./{}/".format(args.floder),"./{}/".format(args.floder),100)
    ssim = compare.calc_ssim("./{}/".format(args.floder),"./{}/".format(args.floder),100)
    psnr = compare.calc_psnr("./{}/".format(args.floder),"./{}/".format(args.floder),100)
    lpip = compare.calc_lpips("./{}/".format(args.floder),"./{}/".format(args.floder),100)
    print("mse2:{},ssim:{},psnr:{},lpip:{}".format(mse,ssim,psnr,lpip))

    mse_score = min(mse_score,mse)
    ssim_score = max(ssim_score,ssim)
    PSNR_score = max(PSNR_score,psnr)
    LPIP_score = min(LPIP_score,lpip)
    print("mse2_:{},ssim_:{},psnr_:{},lpip_:{}".format(mse_score,ssim_score,PSNR_score,LPIP_score))

    adjust_learning_rate(optimizer_gen, iter, args.lr1)
    adjust_learning_rate(optimizer_res_gen, iter, args.lr2)




