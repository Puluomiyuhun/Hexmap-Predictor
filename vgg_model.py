import torch.nn as nn
import torch.nn.functional as F

class VGG16(nn.Module):
 
    def __init__(self,num_classes=30):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128,128,3,padding=1)
        self.conv5 = nn.Conv2d(128,256,3,padding=1)
        self.conv6 = nn.Conv2d(256,256,3,padding=1)
        self.conv7 = nn.Conv2d(256,256,3,padding=1)
        self.conv8 = nn.Conv2d(256,512,3,padding=1)
        self.conv9 = nn.Conv2d(512,512,3,padding=1)
        self.conv10 = nn.Conv2d(512,512,3,padding=1)
        self.conv11 = nn.Conv2d(512,512,3,padding=1)
        self.conv12 = nn.Conv2d(512,512,3,padding=1)
        self.conv13 = nn.Conv2d(512,512,3,padding=1)
 
        self.fc1 = nn.Linear(7*7*512,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,num_classes)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv2(F.relu(self.conv1(x)))),(2,2))
        x = F.max_pool2d(F.relu(self.conv4(F.relu(self.conv3(x)))),(2,2))
        x = F.max_pool2d(F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(x)))))),(2,2))
        x = F.max_pool2d(F.relu(self.conv10(F.relu(self.conv9(F.relu(self.conv8(x)))))),(2,2))
        x = F.max_pool2d(F.relu(self.conv13(F.relu(self.conv12(F.relu(self.conv11(x)))))),(2,2))
        x = F.adaptive_avg_pool2d(x,output_size=(7, 7))        
        x = x.view(x.size(0),-1)
        x = F.dropout(F.relu(self.fc1(x)))
        x = F.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x