import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
import cv2
import numpy as np
from dataset import PredictDataset,Rescale,RandomCrop,ToTensor

class Predictor():
    def __init__(self, model_name, show_compare):
        self.model_name = model_name
        self.show_compare = show_compare
        return
    def predict(self,img_file):
        device = torch.device("cuda")
        net = torch.load('./models/' + self.model_name)
        net.to(device)

        transform = transforms.Compose([
                    Rescale(256),
                    RandomCrop(224),
                    ToTensor()
        ])
        
        transformsed_dataset = PredictDataset(img_file,transform=transform)
        dataloader = DataLoader(transformsed_dataset, batch_size=1,shuffle=False,num_workers=1)

        res = []
        for i, data in enumerate(dataloader,0):
            inputs, labels = data
            inputs = inputs.to(device,dtype=torch.float)
            outputs = net(inputs)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.argmax(outputs)
            res.append(outputs)
            #print(str(outputs)+",", end='')

        img = cv2.imread(img_file)
        height, width, _ = img.shape
        height = height / 48
        width = width / 48

        with open(img_file[0:len(img_file)-4] + "_hex.bmp", "wb") as hexmap:
            hexmap.write(0x424D.to_bytes(2, 'big'))
            lens = 54 + 1024 + 2 + height * width
            hexmap.write(int(lens).to_bytes(2, 'little'))
            hexmap.write(0x0.to_bytes(6, 'big'))
            hexmap.write(0x0436.to_bytes(4, 'little'))
            hexmap.write(0x28.to_bytes(4, 'little'))
            hexmap.write(int(width).to_bytes(4, 'little'))
            
            hexmap.write(int(-height + (1<<32)).to_bytes(4, 'little'))
            hexmap.write(0x1.to_bytes(2, 'little'))
            hexmap.write(0x8.to_bytes(2, 'little'))
            for i in range(0,24):
                hexmap.write(0x0.to_bytes(1, 'little'))
            with open("./dependency/palette.bin", "rb") as palette:
                for i in range(0,1024):
                    bin = palette.read(1)
                    hexmap.write(bin)
            hexmap.write(int(width*3).to_bytes(1, 'little'))
            hexmap.write(int(height*3).to_bytes(1, 'little'))
            for i in range(0,int(height * width)):
                hexmap.write(int(res[i]).to_bytes(1, 'little'))

        split_hex = []
        all_hex = cv2.imread("./dependency/hex.bmp")
        for j in range(0,2):
            for i in range(0,15):
                split_hex.append(all_hex[i*48:(i+1)*48, j*48:(j+1)*48])

        res_hex = np.zeros((int(height) * 48, int(width) * 48, 3),np.uint8)
        for i in range(0,int(height)):
            for j in range(0,int(width)):
                res_hex[i * 48:(i + 1) * 48, j * 48:(j + 1) * 48] = split_hex[res[i * int(width) + j]]
        res_hex = cv2.hconcat([img,res_hex])
        res_hex = cv2.resize(res_hex, None, fx=0.35,fy=0.35)
        if self.show_compare == True:
            cv2.imshow("result", res_hex)
            cv2.waitKey(0)
        cv2.imwrite(img_file[0:len(img_file)-4] + "_compare.bmp", res_hex)