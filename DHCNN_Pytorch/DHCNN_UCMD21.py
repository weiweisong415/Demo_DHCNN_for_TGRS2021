'''
Demo code of our method DHCNN which proposed in TGRS2021
"Deep Hashing Learning for Visual and Semantic Retrieval of Remote Sensing Images"
We careful implement this algorithm with Pytorch and refer to DPSH (https://github.com/jiangqy/DPSH-pytorch).
Here, we would say thanks to these authors.
'''
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import os
import scipy.io
import numpy as np
import pickle
from datetime import datetime

import utils.DataProcessing as DP
import utils.CalcHammingRanking as CalcHR

import CNN_model

def LoadLabel(filename, DATA_DIR):
    path = os.path.join(DATA_DIR, filename)
    fp = open(path, 'r')
    labels = [x.strip() for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int, labels)))

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S

def CreateModel(model_name, bit, use_gpu, no_classes):
    if model_name == 'vgg11':
        vgg11 = models.vgg11(pretrained=True)
        cnn_model = CNN_model.cnn_model(vgg11, model_name, bit, no_classes)
    if model_name == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
        cnn_model = CNN_model.cnn_model(alexnet, model_name, bit, no_classes)
    if use_gpu:
        cnn_model = cnn_model.cuda()
    return cnn_model

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30 ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def GenerateCode(model, data_loader, num_data, bit, use_gpu):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda())
        else: data_input = Variable(data_input)
        output, output_s  = model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B

def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.])))
    return lt
def Totloss(U, B, Sim, lamda, num_train):
    theta = U.mm(U.t()) / 2
    t1 = (theta*theta).sum() / (num_train * num_train)
    l1 = (- theta * Sim + Logtrick(Variable(theta), False).data).sum()
    l2 = (U - B).pow(2).sum()
    l = l1 + lamda * l2
    return l, l1, l2, t1

def DPSH_algo(bit, param, gpu_ind=0):
    # parameters setting
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ind)
    no_classes = 21
    DATA_DIR = 'data/UCMD-21'
    DATABASE_FILE = 'database_index_img.txt'
    TRAIN_FILE = 'train_index_img.txt'
    TEST_FILE = 'test_index_img.txt'

    DATABASE_LABEL = 'database_index_label.txt'
    TRAIN_LABEL = 'train_index_label.txt'
    TEST_LABEL = 'test_index_label.txt'

    batch_size = 50
    epochs = 100
    learning_rate = 0.1
    weight_decay = 10 ** -5
    model_name = 'alexnet'    # vgg11    alexnet
    nclasses = 21
    use_gpu = torch.cuda.is_available()

    filename = param['filename']

    lamda = param['lambda']
    param['bit'] = bit
    param['epochs'] = epochs
    param['learning rate'] = learning_rate
    param['model'] = model_name

    ### data processing
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dset_database = DP.DatasetProcessing(
        DATA_DIR, DATABASE_FILE, DATABASE_LABEL, transformations)

    dset_train = DP.DatasetProcessing(
        DATA_DIR, TRAIN_FILE, TRAIN_LABEL, transformations)

    dset_test = DP.DatasetProcessing(
        DATA_DIR, TEST_FILE, TEST_LABEL, transformations)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

    database_loader = DataLoader(dset_database,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4
                             )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                             )

    test_loader = DataLoader(dset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4
                             )

    ### create model
    model = CreateModel(model_name, bit, use_gpu, no_classes)
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    ### training phase
    # parameters setting
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    B = torch.zeros(num_train, bit)
    U = torch.zeros(num_train, bit)
    train_labels = LoadLabel(TRAIN_LABEL, DATA_DIR)
    train_labels_onehot = EncodingOnehot(train_labels, nclasses)
    test_labels = LoadLabel(TEST_LABEL, DATA_DIR)
    test_labels_onehot = EncodingOnehot(test_labels, nclasses)

    train_loss = []
    map_record = []

    totloss_record = []
    totl1_record = []
    totl2_record = []
    t1_record = []

    Sim = CalcSim(train_labels_onehot, train_labels_onehot)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            train_input, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)
            if use_gpu:
                train_label_onehot = EncodingOnehot(train_label, nclasses)
                train_input, train_label = Variable(train_input.cuda()), Variable(train_label.cuda())
                S = CalcSim(train_label_onehot, train_labels_onehot)
            else:
                train_label_onehot = EncodingOnehot(train_label, nclasses)
                train_input, train_label = Variable(train_input), Variable(train_label)
                S = CalcSim(train_label_onehot, train_labels_onehot)

            model.zero_grad()
            outputs, outputs_s = model(train_input)
            loss_semantic = criterion(outputs_s, train_label)
            for i, ind in enumerate(batch_ind):
                U[ind, :] = outputs.data[i]
                B[ind, :] = torch.sign(outputs.data[i])

            Bbatch = torch.sign(outputs)
            if use_gpu:
                theta_x = outputs.mm(Variable(U.cuda()).t()) / 32
                logloss = (Variable(S.cuda())*theta_x - Logtrick(theta_x, use_gpu)).sum() \
                        / (num_train * len(train_label))
                regterm = (Bbatch-outputs).pow(2).sum() / (num_train * len(train_label))
            else:
                theta_x = outputs.mm(Variable(U).t()) / 2
                logloss = (Variable(S)*theta_x - Logtrick(theta_x, use_gpu)).sum() \
                        / (num_train * len(train_label))
                regterm = (Bbatch-outputs).pow(2).sum() / (num_train * len(train_label))

            loss_p =  - logloss + lamda * regterm
            loss = (1-eta)*loss_p + eta* loss_semantic
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            #print('[Training Phase][Epoch: %3d/%3d][Iteration: %3d/%3d] Loss: %3.5f' % \
            #    (epoch + 1, epochs, iter + 1, np.ceil(num_train / batch_size),loss.data[0]))

        print('[Train Phase][Epoch:%3d/%3d][Loss:%3.5f]' % (epoch+1, epochs, epoch_loss / len(train_loader)), end='')


        optimizer = AdjustLearningRate(optimizer, epoch, learning_rate)

        l, l1, l2, t1 = Totloss(U, B, Sim, lamda, num_train)
        totloss_record.append(l)
        totl1_record.append(l1)
        totl2_record.append(l2)
        t1_record.append(t1)

        #print('[Total Loss: %10.5f][total L1: %10.5f][total L2: %10.5f][norm theta: %3.5f]' % (l, l1, l2, t1), end='')
        #print('[Total Loss: %10.5f]' % l, end='')
        print('[Total Loss: %10.5f]' % l)
        ### testing during epoch
        if epoch%10==0:
            qB = GenerateCode(model, test_loader, num_test, bit, use_gpu)
            tB = torch.sign(B).numpy()
            map_ = CalcHR.CalcMap(qB, tB, test_labels_onehot.numpy(), train_labels_onehot.numpy())
        #train_loss.append(epoch_loss / len(train_loader))
        #map_record.append(map_)

            print('[Test Phase ] MAP(retrieval train): %3.5f' % ( map_))
        #print(len(train_loader))
    ### evaluation phase
    ## create binary code
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

    start.record()
    model.eval()
    database_labels = LoadLabel(DATABASE_LABEL, DATA_DIR)
    database_labels_onehot = EncodingOnehot(database_labels, nclasses)
    qB = GenerateCode(model, test_loader, num_test, bit, use_gpu)
    dB = GenerateCode(model, database_loader, num_database, bit, use_gpu)

   # scipy.io.savemat('./data/UCMD-21/DHNN-64-bits.mat', mdict={'database_bit': dB, 'test_bit': qB})
    map = CalcHR.CalcMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy())
    print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))
    result = {}
    result['qB'] = qB
    result['dB'] = dB
    result['train loss'] = train_loss
    result['map record'] = map_record
    result['map'] = map
    result['param'] = param
    result['total loss'] = totloss_record
    result['l1 loss'] = totl1_record
    result['l2 loss'] = totl2_record
    result['norm theta'] = t1_record
    result['filename'] = filename

    return result

if __name__=='__main__':
    bit = 16
    lamda = 10  # approximate parameter
    eta = 0.2  # balanced parameter
    gpu_ind = 0
    filename = 'log/DHCNN_' + str(bit) + 'bits_UCMD-21_' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.pkl'
    param = {}
    param['lambda'] = lamda
    param['filename'] = filename
    result = DPSH_algo(bit, param, gpu_ind)
    fp = open(result['filename'], 'wb')
    pickle.dump(result, fp)
    fp.close()

