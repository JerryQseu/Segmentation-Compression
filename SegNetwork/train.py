import torch
import read as data
import torch.utils.data as Datas
import CCNet as Network
import metrics as criterion

from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device_ids = [0]

# model = Module()

device = torch.device("cuda:0")
data = data.train_data

dataloder = Datas.DataLoader(dataset=data,batch_size=1,shuffle=True)

#
fusenet = Network.SegNetwork().to(device)
opt = torch.optim.Adam(fusenet.parameters(),lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-5)
#######
# pretrained_dict = torch.load('./pkl/net_epoch_19-fuseNetwork.pkl')
# model_dict = fusenet.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# fusenet.load_state_dict(model_dict)


###
criterion_L1 = torch.nn.L1Loss()
criterion_MSE = torch.nn.MSELoss()
criterion_BCE = torch.nn.BCEWithLogitsLoss()
criterion_CE = criterion.crossentry()
criterion_ncc = criterion.NCC().loss
criterion_grad = criterion.Grad('l2',4).loss
criterion_dice = criterion.DiceMeanLoss()
criterion_kl = torch.nn.KLDivLoss(reduction='mean')
criterion_dice1 = criterion.DiceMeanLoss1()

fake_A_buffer = criterion.ReplayBuffer()
fake_B_buffer = criterion.ReplayBuffer()


for epoch in range(200):
    meansegdice = 0
    for step, (img, label) in enumerate(dataloder):

        img = img.to(device).float()
        label = label.to(device).float()
        b, c, w, h = img.shape



        segresult = fusenet(img)

        lossseg_ed_es = criterion_dice1(segresult, label)
        lossseg_ce = criterion_CE(segresult, label)

        loss = lossseg_ed_es + lossseg_ce
        opt.zero_grad()
        loss.backward()
        opt.step()


        if step % 2 == 0:
            torch.save(fusenet.state_dict(), './pkl/net_epoch_' + str(epoch) + '-fuseNetwork.pkl')


        meansegdice += lossseg_ed_es.data.cpu().numpy()

        print('EPOCH:', epoch, '|Step:', step,
              '|loss_seg:', loss.data.cpu().numpy(),'|lossseg_ce:', lossseg_ce.data.cpu().numpy(),'|lossseg_f1:', lossseg_ed_es.data.cpu().numpy())

    print('epoch', epoch, '|meansegdice:',(meansegdice / step))

