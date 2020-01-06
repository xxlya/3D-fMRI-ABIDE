'''
GitHub - kenshohara/3D-ResNets-PyTorch: 3D ResNets for Action Recognition (CVPR 2018)
'''
from __future__ import print_function
import os
import sys
import json
import h5py
import numpy as np
import deepdish as dd
from scipy.io import loadmat
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from opts import parse_opts
from model.net import resnet10
from model.data_loader_crnn import fMRIDataset_CRNN,get_loader
from model.data_loader_2c import fMRIDataset_2C
from model.data_loader import fMRIDataset
from utils import Logger,perf_measure
from train import train_epoch
from validation import val_epoch
import test
import pandas as pd
from model.C3D import MyNet
from model.CNN_LSTM import CNN_LSTM
from torch.autograd import Variable
import os
import copy
os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.manual_seed(666)
def main():

    print(torch.version.cuda)
    opt = parse_opts()
    print(opt)
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    device = torch.device("cuda" if opt.use_cuda else "cpu")

    # Read Phenotype
    csv = pd.read_csv(opt.csv_dir)

    if opt.cross_val:

        for fold in range(5):
            train_ID = dd.io.load(os.path.join(opt.MAT_dir, 'fold_split_rep.h5'))[fold]['X_train']
            val_ID = dd.io.load(os.path.join(opt.MAT_dir, 'fold_split_rep.h5'))[fold]['X_test']

            # ==========================================================================#
            #                       1. Network Initialization                          #
            # ==========================================================================#
            torch.manual_seed(opt.manual_seed)
            if opt.architecture == 'ResNet':
                kwargs = {'inchn': opt.win_size, 'sample_size': opt.sample_size, 'sample_duration': opt.sample_duration,
                          'num_classes': opt.n_classes}
                model = resnet10(**kwargs).to(device)
            elif opt.architecture == 'NC3D':
                model = MyNet(opt.win_size, opt.nb_filter, opt.batch_size).to(device)
            elif opt.architecture == 'CRNN':
                model = CNN_LSTM(opt.win_size, opt.nb_filter, opt.batch_size, opt.sample_size, opt.sample_duration, opt.rep).to(device)
            else:
                print('Architecture is not available.')
                raise LookupError
            print(model)
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            num_params = sum([np.prod(p.size()) for p in model_parameters])
            print('number of trainable parameters:', num_params)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            class_weights = torch.FloatTensor(opt.weights).to(device)
            criterion = nn.CrossEntropyLoss(weight= class_weights)
            criterion.to(device)

            # ==========================================================================#
            #                     2. Setup Dataloading Paramters                       #
            # ==========================================================================#
            '''load subjects ID'''
            ID = csv['SUB_ID'].values
            win_size = opt.win_size  # num of channel input
            T = opt.sample_duration  # total length of fMRI
            num_rep = T // win_size  # num of repeat the ID

            # ==========================================================================#
            #                     3. Training and Validation                            #
            # ==========================================================================#

            if opt.architecture == 'ResNet':
                training_data = fMRIDataset(opt.datadir, win_size,  train_ID, T, csv)
            elif opt.architecture == 'NC3D':
                training_data = fMRIDataset_2C(opt.datadir, train_ID)
            elif opt.architecture == 'CRNN':
                training_data = fMRIDataset_CRNN(opt.datadir, win_size, train_ID, T, csv)
            else:
                print('Architecture is not available.')
                raise LookupError
            train_loader = torch.utils.data.DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=False)
            log_path = os.path.join(opt.result_path, str(fold))
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            train_logger = Logger(
                os.path.join(log_path, 'train.log'),
                ['epoch', 'loss', 'acc', 'lr'])
            train_batch_logger = Logger(
                os.path.join(log_path, 'train_batch.log'),
                ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

            '''optimization'''
            if opt.nesterov:
                dampening = 0
            else:
                dampening = opt.dampening
            if opt.optimizer == 'sgd':
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=opt.nesterov)
            elif opt.optimizer == 'adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=opt.learning_rate,
                    weight_decay=opt.weight_decay
                )
            elif opt.optimizer == 'adadelta':
                optimizer = optim.Adadelta(
                    model.parameters(),
                    lr=opt.learning_rate,
                    weight_decay=opt.weight_decay
                )
            # scheduler = lr_scheduler.ReduceLROnPlateau(
            #     optimizer, 'min', patience=opt.lr_patience)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
            if not opt.no_val:
                if opt.architecture == 'ResNet':
                    validation_data = fMRIDataset(opt.datadir, win_size, val_ID, T, csv)
                elif opt.architecture == 'NC3D':
                    validation_data = fMRIDataset_2C(opt.datadir, val_ID)
                elif opt.architecture == 'CRNN':
                    validation_data = fMRIDataset_CRNN(opt.datadir, win_size, val_ID, T, csv)
                val_loader = torch.utils.data.DataLoader(
                    validation_data,
                    batch_size=opt.n_val_samples,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=False)
                val_logger = Logger(
                    os.path.join(log_path, 'val.log'), ['epoch', 'loss', 'acc'])

            if opt.resume_path:
                print('loading checkpoint {}'.format(opt.resume_path))
                checkpoint = torch.load(opt.resume_path)
                # assert opt.arch == checkpoint['arch']

                opt.begin_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                if not opt.no_train:
                    optimizer.load_state_dict(checkpoint['optimizer'])

            print('run')
            best_loss = 1e4
            for i in range(opt.begin_epoch, opt.n_epochs + 1):
                if not opt.no_train:
                    train_epoch(i, train_loader, model, criterion, optimizer, opt, log_path,
                                train_logger, train_batch_logger)
                if not opt.no_val:
                    validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                                val_logger)
                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(best_model_wts, os.path.join(log_path, str(fold) + '_best.pth'))
                if not opt.no_train and not opt.no_val:
                    #scheduler.step(validation_loss)
                    scheduler.step()

                model_wts = copy.deepcopy(model.state_dict())
                torch.save(model_wts, os.path.join(log_path, str(fold) +'_epoch_' +str(i)+ '.pth'))


            # =========================================================================#
            #                            4. Testing                                    #
            # =========================================================================#

            if opt.test:
                model = MyNet(opt.win_size, opt.nb_filter, opt.batch_size).to(device)
                model.load_state_dict(torch.load(os.path.join(log_path, str(fold) + '_best.pth')))
                test_details_logger = Logger(os.path.join(log_path, 'test_details.log'),['sub_id','pos','neg'])
                test_logger = Logger(os.path.join(log_path, 'test.log'), ['fold', 'real_Y', 'pred_Y', 'acc', 'sen','spec','ppv','npv'])
                real_Y = []
                pred_Y = []
                model.eval()
                test_loader = torch.utils.data.DataLoader(
                    validation_data,
                    batch_size=142,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=False)
                with torch.no_grad():
                    for i, (inputs, targets) in enumerate(test_loader):
                        real_Y.append(targets[0])
                        inputs, targets = inputs.to(device), targets.to(device)
                        inputs = Variable(inputs).float()
                        targets = Variable(targets).long()
                        outputs = model(inputs)
                        rest = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                        pos = np.sum(rest == targets.detach().cpu().numpy())
                        neg = len(rest) - pos
                        print('pos:', pos, '  and neg:', neg)
                        test_details_logger.log({'sub_id': val_ID[i*142], 'pos': pos, 'neg': neg})
                        if np.sum(rest==1) >= np.sum(rest==0):
                            pred_Y.append(1)
                        else:
                            pred_Y.append(0)
                TP, FP, TN, FN = perf_measure(real_Y, pred_Y )
                acc = (TP+TN)/(TP+TN+FP+FN)
                sen = TP/(TP+FN)
                spec = TN/(TN+FP)
                ppv = TP/(TP+FP)
                npv = TN/(TN+FN)
                test_logger.log({'fold':fold,'real_Y':real_Y,'pred_Y':pred_Y,'acc':acc,'sen':sen,'spec':spec,'ppv':ppv,'npv':npv})

    else:

        fold = opt.fold
        train_ID =  dd.io.load(os.path.join(opt.MAT_dir, 'fold_split_rep.h5'))[fold]['X_train']
        val_ID = dd.io.load(os.path.join(opt.MAT_dir, 'fold_split_rep.h5'))[fold]['X_test']

        # ==========================================================================#
        #                       1. Network Initialization                          #
        # ==========================================================================#
        torch.manual_seed(opt.manual_seed)
        if opt.architecture == 'ResNet':
            kwargs = {'inchn': opt.win_size, 'sample_size': opt.sample_size, 'sample_duration': opt.sample_duration,
                      'num_classes': opt.n_classes}
            model = resnet10(**kwargs).to(device)
        elif opt.architecture == 'NC3D':
            model = MyNet(opt.win_size, opt.nb_filter, opt.batch_size).to(device)
        elif opt.architecture == 'CRNN':
            model = CNN_LSTM(opt.win_size, opt.nb_filter, opt.batch_size, opt.s_sz, opt.sample_duration, opt.rep).to(device)
        else:
            print('Architecture is not available.')
            raise LookupError
        print(model)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum([np.prod(p.size()) for p in model_parameters])
        print('number of trainable parameters:', num_params)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = torch.FloatTensor(opt.weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion.to(device)

        # ==========================================================================#
        #                     2. Setup Dataloading Paramters                       #
        # ==========================================================================#
        '''load subjects ID'''
        win_size = opt.win_size  # num of channel input
        T = opt.sample_duration  # total length of fMRI

        # ==========================================================================#
        #                     3. Training and Validation                            #
        # ==========================================================================#
         # repeat the ID, in order to visit all the volumes in fMRI, this will be input to the dataloader
        if opt.architecture == 'ResNet':
            training_data = fMRIDataset(opt.datadir, opt.s_sz, train_ID, T, csv, opt.rep)
        elif opt.architecture == 'NC3D':
            training_data = fMRIDataset_2C(opt.datadir,train_ID)
        elif opt.architecture == 'CRNN':
            training_data = fMRIDataset_CRNN(opt.datadir, opt.s_sz, train_ID, T, csv, opt.rep)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            #num_workers=opt.n_threads,
            pin_memory=True)
        log_path = opt.result_path
        print('log_path',log_path)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        train_logger = Logger(
            os.path.join(log_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(log_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        '''optimization'''
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        if opt.optimizer =='sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
        elif opt.optimizer =='adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=opt.learning_rate,
                weight_decay=opt.weight_decay
                )
        elif opt.optimizer =='adadelta':
            optimizer = optim.Adadelta(
                model.parameters(),
                lr=opt.learning_rate,
                weight_decay=opt.weight_decay
            )
        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 'min', patience=opt.lr_patience)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        if not opt.no_val:
            if opt.architecture == 'ResNet':
                validation_data = fMRIDataset(opt.datadir, opt.s_sz, val_ID, T, csv, opt.rep)
            elif opt.architecture == 'NC3D':
                validation_data = fMRIDataset_2C(opt.datadir, val_ID)
            elif opt.architecture == 'CRNN':
                validation_data = fMRIDataset_CRNN(opt.datadir, opt.s_sz, val_ID, T, csv, opt.rep)
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.n_val_samples,
                shuffle=False,
                #num_workers=opt.n_threads,
                pin_memory=True)
            val_logger = Logger(
                os.path.join(log_path, 'val.log'), ['epoch', 'loss', 'acc'])

        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            # assert opt.arch == checkpoint['arch']

            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            if not opt.no_train:
                optimizer.load_state_dict(checkpoint['optimizer'])

        print('run')
        for i in range(opt.begin_epoch, opt.n_epochs + 1):
            if not opt.no_train:
                train_epoch(i, train_loader, model, criterion, optimizer, opt, log_path,
                            train_logger, train_batch_logger)
            if not opt.no_val: # when epoch is greater then 5, we start to do validation
                validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger)

            if not opt.no_train and not opt.no_val:
                scheduler.step(validation_loss)

        # =========================================================================#
        #                            4. Testing                                    #
        # =========================================================================#

        if opt.test:
            test_details_logger = Logger(os.path.join(opt.result_path, 'test_details.log'), ['sub_id', 'pos', 'neg'])
            test_logger = Logger(os.path.join(opt.result_path, 'test.log'),
                                 ['fold', 'real_Y', 'pred_Y', 'acc', 'sen', 'spec', 'ppv', 'npv'])
            real_Y = []
            pred_Y = []
            model.eval()
            test_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=142,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=False)
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(test_loader):
                    real_Y.append(targets)
                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs = Variable(inputs).float()
                    targets = Variable(targets).long()
                    outputs = model(inputs)
                    rest = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                    pred_Y.append(outputs.detach().cpu().numpu())
                    pos = np.sum(rest == targets.detach().cpu().numpu())
                    neg = len(rest) - pos
                    print('pos:', pos, '  and neg:', neg)
                    test_details_logger.log({'sub_id': val_ID[i * 142], 'pos': pos, 'neg': neg})
            TP, FP, TN, FN = perf_measure(real_Y, pred_Y)
            acc = (TP + TN) / (TP + TN + FP + FN)
            sen = TP / (TP + FN)
            spec = TN / (TN + FP)
            ppv = TP / (TP + FP)
            npv = TN / (TN + FN)
            test_logger.log(
                {'fold': fold, 'real_Y': real_Y, 'pred_Y': pred_Y, 'acc': acc, 'sen': sen, 'spec': spec, 'ppv': ppv,
                 'npv': npv})


if __name__ == '__main__':
    main()
