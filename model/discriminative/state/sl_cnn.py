import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from model.discriminative.energy.utils import *
import matplotlib.pyplot as plt
import copy
import os
import time
from time import perf_counter
import argparse
import random

def main():
    seed_number = 1234
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)

    parser = argparse.ArgumentParser()
    parser.add_argument('-lrIndx', type=int, help='Input number')
    args = parser.parse_args() # args.lrIndx
    lrIndx = args.lrIndx
    lrs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6]
    lr = lrs[lrIndx]

    print(f"device is gpu: {torch.cuda.is_available()}")
    print(torch.cuda.device_count())
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    start_time = perf_counter()
    ## ------------------params to modify-------------------------- ##
    # -------------system----------------
    L = 4 # length of the lattice
    dim = 2 # dimension of the lattice
    
    T_min, T_max = 0.1, 5.1
    dT = 0.1
    nt = 1 + np.int64(np.round(np.round((T_max-T_min) / dT)))
    T = np.linspace(T_min, T_max, nt)
    T = np.round(T, 2)
    
    eqSteps = 10**5 # mixing time
    mcSteps = 10**6 # MC steps = number of samples
    sampling_interval = 10 # sampling interval
    
    # -------------training params----------------
    epochs = 100
    
    # binary cross entropy loss
    def loss_func(outputs, target):
        pred = outputs[:,0]
        target = torch.reshape(target, (-1,))
        loss1 = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
        return loss1
        
    for num_samples in [10**1, 3*10**1, 10**2, 3*10**2, 10**3, 3*10**3, 10**4, 3*10**4, 10**5, 3*10**5, 10**6]:
        batch_size = 64 if num_samples > 100 else num_samples
        epoch_checkpoint = int(epochs/10)
            
        print(f"-------------------num_samples={num_samples}-------------------")

        dataload_folder = f"../../../data/L={L}_Tmin={np.round(T_min,1)}_Tmax={np.round(T_max,1)}_eqSteps={eqSteps}_mcSteps={mcSteps}_interval={sampling_interval}/state"
        
        spins_data = np.zeros((nt, num_samples, L, L), dtype=np.float32)

        for tt in tqdm(range(nt)):
            spins_data[tt,:,:,:] = np.load(f"{dataload_folder}/T={T[tt]}.npy")[:num_samples, :].astype(np.float32).reshape((num_samples, L, L))
    
        flip = np.random.randint(0, 2, (nt, num_samples, 1, 1)) * 2 - 1
        spins_data *= flip
    
        train_data = np.zeros((2, num_samples, L, L), dtype=np.float32)
        data_mean, data_std = np.mean(spins_data, axis=(0,1)), np.std(spins_data, axis=(0,1))

        train_data[0,:,:,:] = spins_data[0,:,:,:]
        train_data[-1,:,:,:] = spins_data[-1,:,:,:]
        train_data = np.reshape(train_data, (-1, L, L))
        train_data = (train_data - data_mean) / data_std
        train_data = torch.tensor(train_data).unsqueeze(1)
    
        data = (spins_data - data_mean) / data_std
    
        targets = torch.ones(num_samples*2)
        targets[num_samples:] = 0
        targets = targets.unsqueeze(1)

        dataset = TensorDataset(train_data, targets)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True, drop_last=True)
    
        start = perf_counter()
        print(f"-------------------lr={lr}, epochs={epochs}-------------------")
        model = CNN_SL().to(device)
        num_params = get_n_params(model)
        print(f"Number of parameters: {num_params}")
    
        optimizer = optim.Adam(model.parameters(), lr=lr)

        losses = torch.zeros(epochs,dtype=torch.float32).to(device)

        # Training loop
        model.train()
        for epoch in tqdm(range(epochs)):
            running_loss = 0.0
            num_batches = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
    
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                num_batches += 1
            losses[epoch] = running_loss / num_batches
            if (epoch+1) % epoch_checkpoint == 0:
                print(f'Epoch {epoch}: Average Loss = {losses[epoch]}')
            os.makedirs(f"./trained/CNN_SL/numsample={num_samples}_lr={lr}_epochs={epochs}/", exist_ok=True)
            torch.save(model.state_dict(), f"./trained/CNN_SL/numsample={num_samples}_lr={lr}_epochs={epochs}/epoch={epoch}.pt")

        end = time.perf_counter()
        print(f"eplased time = {(end-start)}s")
        print(f'\n---------------------***save file***------------------------\n')
        losses_cpu = losses.cpu().numpy()
        np.save(f"./trained/CNN_SL/numsample={num_samples}_lr={lr}_epochs={epochs}/loss.npy", losses_cpu)   

    elapsed_time = perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()