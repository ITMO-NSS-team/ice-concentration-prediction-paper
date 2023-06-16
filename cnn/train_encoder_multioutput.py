import os
import pickle
from datetime import datetime
from pytorch_msssim import ssim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from EncoderForecasterBase import EncoderForecasterBase
from TensorBuilder import multioutput_tensor

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')
batch_size = 10
epochs = 500
learning_rate = 1e-3

data_freq = 7

x_virg = []
temp_ar = []
for file in os.listdir('../../matrices/osisaf/train'):
    date = datetime.strptime(file, 'osi_iceconc_%Y%m%d.npy')
    if date.year < 2013:
        array = np.load(f'../../matrices/osisaf/train/{file}')
        temp_ar.append(array)
        if len(temp_ar) == data_freq:
            temp_ar = np.array(temp_ar)
            temp_ar = temp_ar[-1]
            x_virg.append(temp_ar)
            temp_ar = []
for file in os.listdir('../../matrices/osisaf/test'):
    date = datetime.strptime(file, 'osi_iceconc_%Y%m%d.npy')
    if date.year < 2013:
        array = np.load(f'../../matrices/osisaf/test/{file}')
        temp_ar.append(array)
        if len(temp_ar) == data_freq:
            temp_ar = np.array(temp_ar)
            temp_ar = temp_ar[-1]
            x_virg.append(temp_ar)
            temp_ar = []
x_virg = np.array(x_virg)

pre_history_size = 104
forecast_size = 52

dataset = multioutput_tensor(pre_history_size, forecast_size, x_virg)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print('Loader created')

encoder = EncoderForecasterBase()
encoder.init_encoder(input_size=[125, 125],
                     n_layers=5,
                     in_channels=pre_history_size,
                     out_channels=forecast_size)
encoder.to(device)
print(encoder)

optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

for epoch in range(epochs):
    loss = 0
    for train_features, test_features in dataloader:
        train_features = train_features.to(device)
        test_features = test_features.to(device)
        optimizer.zero_grad()
        outputs = encoder(train_features)
        ssim_loss = 1 - ssim(outputs, test_features, data_range=1, size_average=True)
        l1_loss = criterion(outputs, test_features)
        #train_loss = ssim_loss+l1_loss
        train_loss = l1_loss
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader)

    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

'''with open(f"../../fitted_models/long_term_multioutput/osi/osi_1990-2013_lag{pre_history_size}_for{forecast_size}w.pkl", "wb") as fp:
    pickle.dump(encoder, fp)'''

save_path = f'model_weights_104_52_1990-2013.pt'
torch.save(encoder.state_dict(), save_path)