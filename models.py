import torch
import torch.nn as nn
import torch.nn.functional as F
from metadatanorm import MetadataNorm

class BaselineNet(nn.Module):
    def __init__(self):
        """ Baseline CNN model with 2 convolutional layers and 2 linear layers. """
     
        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(18432, 84)
        self.fc2 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 18432)
        x = self.fc1(x)   
        fc = x.cpu().detach().numpy()
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x, fc

class MDN_Linear(nn.Module):
    def __init__(self, dataset_size, batch_size, kernel):
        """ MDN-Linear model: Baseline CNN model with 2 convolutional and 2 linear layers with MDN applied
            to the last linear layer before the output layer. 
        Args:
          dataset_size (int): size of dataset
          batch_size (int): batch size
          kernel (2d vector): precalculated kernel for MDN based on the vector X of confounders (X^TX)^-1.
              kernel needs to be set before training, and cfs needs to be set during training for each batch.
        """
        
        super(MDN_Linear, self).__init__()
        self.N = batch_size
        self.C = kernel.shape[0] 
        self.kernel = kernel
        self.cfs = nn.Parameter(torch.randn(batch_size, self.C), requires_grad=False)
        self.dataset_size = dataset_size
         
        # Convolutional and MDN layers
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(18432, 84)
        self.metadatanorm = MetadataNorm(self.N, self.kernel, self.dataset_size, 84)
        self.fc2 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(-1, 18432)
        x = self.fc1(x)   
        self.metadatanorm.cfs = self.cfs
        x = self.metadatanorm(x)
        fc = x.cpu().detach().numpy()
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x, fc
    
class MDN_Conv(nn.Module):
    def __init__(self, dataset_size, batch_size, kernel):
        """ MDN-Conv model: Baseline CNN model with 2 convolutional and 2 linear layers with MDN applied
            to every convolutional layer and the last linear layer before the output layer.
        Args:
          dataset_size (int): size of dataset
          batch_size (int): batch size
          kernel (2d vector): precalculated kernel for MDN based on the vector X of confounders (X^TX)^-1.
              kernel needs to be set before training, and cfs needs to be set during training for each batch.
        """
        
        super(MDN_Conv, self).__init__()
        self.N = batch_size
        self.C = kernel.shape[0] 
        self.kernel = kernel
        self.cfs = nn.Parameter(torch.randn(batch_size, self.C), requires_grad=False)
        self.dataset_size = dataset_size
 
        # Convolutional and MDN layers
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.metadatanorm1 = MetadataNorm(self.N, self.kernel, self.dataset_size, 16*28*28)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.metadatanorm2 = MetadataNorm(self.N, self.kernel, self.dataset_size, 32*24*24)
        self.fc1 = nn.Linear(18432, 84)
        self.metadatanorm3 = MetadataNorm(self.N, self.kernel, self.dataset_size, 84)
        self.fc2 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        self.metadatanorm1.cfs = self.cfs
        x = self.metadatanorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        self.metadatanorm2.cfs = self.cfs
        x = self.metadatanorm2(x) 
        x = F.relu(x)
        x = x.view(-1, 18432)
        x = self.fc1(x)   
        self.metadatanorm3.cfs = self.cfs
        x = self.metadatanorm3(x)
        fc = x.cpu().detach().numpy()
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x, fc
