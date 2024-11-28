import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def mse(x, y):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.mean((x - y) ** 2)
    else:
        return torch.mean((x - y) ** 2)
    
def sl(model, data, nt, dT, device):
    xs, ys = np.zeros(nt), np.zeros(nt)
    r = np.zeros(nt)
    model.eval()
    if isinstance(model, CNN_SL):
        for tt in range(nt):
            with torch.no_grad():
                outputs = F.softmax(model(torch.tensor(data[tt,:,:,:]).unsqueeze(1).to(device)), dim=1)
                # print(outputs)
                outputs = outputs.cpu().numpy()
                xs[tt] = np.mean(outputs[:,0])
                ys[tt] = 1.0 - xs[tt]
    else:
        for tt in range(nt):
            with torch.no_grad():
                outputs = F.softmax(model(torch.tensor(data[tt,:]).to(device)), dim=1)
                outputs = outputs.cpu().numpy()
                xs[tt] = np.mean(outputs[:,0])
                ys[tt] = 1.0 - xs[tt]
    for tt in range(1,nt-1):
        r[tt] = (abs(xs[tt+1] - xs[tt-1]) + abs(ys[tt+1] - ys[tt-1])) / (2 * dT)/2
    return r[1:-1]
    
def lbc(models, data, num_samples, device, n = 5):
    nt = len(models)-1
    r = np.zeros(nt+1)
    for bp in range(1,nt):
        model = models[bp]
        model.eval()
        with torch.no_grad():
            bl = np.maximum(bp-n, 0)
            br = np.minimum(nt, bp+n)
            output1 = F.softmax(model(data[bl*num_samples:bp*num_samples].to(device)), dim=1)
            output2 = F.softmax(model(data[bp*num_samples:br*num_samples].to(device)), dim=1)
            err1 = torch.min(output1, 1)[0]
            err2 = torch.min(output2, 1)[0]
            err1 = err1.reshape(num_samples, -1)
            err2 = err2.reshape(num_samples, -1)
            err = (torch.sum(torch.mean(err1, 0))/(bp-bl) + torch.sum(torch.mean(err2, 0))/(br-bp))/2
            r[bp] = 1- 2 * err
            # r[bp] = (torch.min(output1, 1)[0].mean() + torch.min(output2, 1)[0].mean()) / 2
    return r[1:-1]

def pbm(model, data, num_samples,nt, dT, device):
    mus, sigmas = torch.zeros(nt), torch.zeros(nt)
    r = np.zeros(nt)
    model.eval()
    for tt in range(nt):
        with torch.no_grad():
            outputs = model(torch.tensor(data[tt*num_samples:(tt+1)*num_samples,:,:,:]).to(device))
            mus[tt] = torch.mean(outputs)
            sigmas[tt] = torch.std(outputs)
    mus = mus.cpu().numpy()
    sigmas = sigmas.cpu().numpy()
    for tt in range(1,nt-1):
        r[tt] = (mus[tt+1] - mus[tt-1]) / (2*dT)
        r[tt] = r[tt] / sigmas[tt] if sigmas[tt] > 1e-7 else 0.0
    return np.abs(r[1:-1])
    

class CNN_SL(nn.Module):
    def __init__(self, hidden_channels=10, hidden_dims=10):
        super(CNN_SL, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, 3, padding=2, padding_mode='circular')
        self.ln = nn.LayerNorm(6)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_channels * 6 * 6, hidden_dims)
        self.ln1 = nn.LayerNorm(hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.ln2 = nn.LayerNorm(hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 2)

    def forward(self, x):
        x = self.conv1(x)  
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
        
class CNN_LBC(nn.Module):
    def __init__(self, hidden_channels=10, hidden_dims=10):
        super(CNN_LBC, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, 3, padding=2, padding_mode='circular')
        self.ln = nn.LayerNorm(6)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_channels * 6 * 6, hidden_dims)
        self.ln1 = nn.LayerNorm(hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.ln2 = nn.LayerNorm(hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 2)

    def forward(self, x):
        x = self.conv1(x)  
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
        
class CNN_PBM(nn.Module):
    def __init__(self,hidden_channels=10, hidden_dims=10):
        super(CNN_PBM, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_channels, 3, padding=2, padding_mode='circular')
        self.ln = nn.LayerNorm(6)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_channels * 6 * 6, hidden_dims)
        self.ln1 = nn.LayerNorm(hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.ln2 = nn.LayerNorm(hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 1)

    def forward(self, x):
        x = self.conv1(x)  
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x