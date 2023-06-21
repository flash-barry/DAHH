from torch.utils.data import DataLoader
from torch import optim
from models.DAHH import DAHH, JointLoss, LblPred
import torch
from models.data import UnitData
from rich.progress import track
import pandas as pd
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.0001
milestones = [10, 40, 60, 80, 100]
gamma = 0.5
epoch_num = 500


def train():
    model = DAHH()
    if torch.cuda.is_available():
        model = model.to(device)

    unit_data = UnitData(True)
    dataloader = DataLoader(unit_data, batch_size=1, shuffle=False)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    criterion = JointLoss()

    for epoch in range(epoch_num):
        model.train()
        criterion.train()

        train_loss = 0.
        train_iter_num = 0

        for X, H, gamma_2, gamma_1, label2, label1 in track(dataloader, description='training'):
            X = X.to(device).squeeze(0)
            H = H.to(device).squeeze(0)
            gamma_2 = gamma_2.to(device).squeeze(0)
            gamma_1 = gamma_1.to(device).squeeze(0)
            label2 = label2.to(device).squeeze(0)
            label1 = label1.to(device).squeeze(0)

            optimizer.zero_grad()
            output1, output2 = model(H, X, gamma_2, gamma_1)
            loss = criterion(output1, label1.long(), output2, label2.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iter_num += 1

        scheduler.step()
        print(epoch, 'Train Loss=%.4f' % (train_loss / train_iter_num))
        
        if (epoch+1)%15 == 0:
            torch.save(model, 'xxx'.format(epoch+1))


def predict():
    S = np.load('epoch_128.npy')
    Phi = np.load('Phi.npy')
    S = torch.Tensor(S).to(device)
    Phi = torch.Tensor(Phi).to(device)

    predicter = LblPred(S, Phi)
    predicter = predicter.to(device)

    unit_data = UnitData(True)
    dataloader = DataLoader(unit_data, batch_size=1, shuffle=False)
    for x, _, _, _, _, _ in dataloader:
        label1 = []
        label2 = []
        x = x.squeeze(0).to(device)
        res1, res2 = predicter(x)

        label1.append(int(res1.max(1)[1]))
        label2.append(int(res2.max(1)[1]))
        res = {"label1": label1, "label2": label2}
        df = pd.DataFrame(res)
        df.to_csv('{}.tsv'.format(index))
        print('predict finish')


if __name__ == '__main__':
    train()
    # predict()

