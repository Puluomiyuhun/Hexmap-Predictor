import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
import cv2
import vgg_model
from dataset import CatDogDataset,Rescale,RandomCrop,ToTensor
import argparse
 
##自定义数据集加载和数据处理类

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=100, help='单批batch的图片总数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据集的workders数量')
    parser.add_argument('--epoch_num', type=int, default=50, help='训练总轮数')
    parser.add_argument('--batch_print', type=int, default=200, help='每训练多少个batch，打印一次loss')
    parser.add_argument('--epoch_save', type=int, default=200, help='每训练多少个轮，保存一次模型权重')
    parser.add_argument('--resume_train', type=str, default='', help='如果要继续训练，填写权重的路径')
    args = parser.parse_args()

    transform = transforms.Compose([
                Rescale(256),
                RandomCrop(224),
                ToTensor()
    ])
    
    transformsed_dataset = CatDogDataset('training_data/train.csv',transform=transform)
    dataloader = DataLoader(transformsed_dataset, batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    print("The data is successfully loaded.")
    
    ####定义指定ＧＰＵ设备
    device = torch.device("cuda")
    print("The device is on " + str(device) + ".")
    
    if args.resume_train == '':
        net = vgg_model.VGG16()
    else:
        net = torch.load(args.resume_train)
    net.to(device)
    print("the VGG network is constructed.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
    
    print("Begin Training!")
    for epoch in range(args.epoch_num):
        running_loss = 0.0
        for i, data in enumerate(dataloader,0):
            inputs, labels = data
            inputs = inputs.to(device,dtype=torch.float)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if i % args.batch_print == args.batch_print - 1:
                print('[%d,%5d] loss: %.3f'%(epoch +1, i+1, running_loss/200))
                running_loss = 0.0
        if epoch % args.epoch_save == 0:
            torch.save(net,'./models/' + str(epoch) + '.pt')
    print('Finished Training')
    #torch.save(net,'./models/final.pt')