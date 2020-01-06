import torch
import model.cnn_rnn.convolutional_rnn as cf
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class CNN_LSTM(nn.Module):
    def __init__(self,window,nb_filter,batch_sz,s_sz,T,rep):
        super(CNN_LSTM,self).__init__()
        self.window = window
        self.nb_filter = nb_filter
        self.b_sz = batch_sz * rep
        self.s_sz = s_sz
        self.T = T


        self.lstm1 = cf.Conv3dPeepholeLSTM(1,  # Corresponds to input size
                                           out_channels= self.nb_filter[0] ,  # Corresponds to hidden size
                                           kernel_size=(5, 5, 5),  # Int or List[int]
                                           num_layers=1,
                                           bidirectional= False,
                                           dilation=2, stride=1, dropout=0.5)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.lstm2 = cf.Conv3dPeepholeLSTM(self.nb_filter[0],  # Corresponds to input size
                                           out_channels= self.nb_filter[1] ,  # Corresponds to hidden size
                                           kernel_size=(3, 3, 3),  # Int or List[int]
                                           num_layers=1,
                                           bidirectional= False,
                                           dilation=2, stride=1, dropout=0.5)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.lstm3 = cf.Conv3dPeepholeLSTM(self.nb_filter[1],  # Corresponds to input size
                                           out_channels= self.nb_filter[2] ,  # Corresponds to hidden size
                                           kernel_size=(3, 3, 3),  # Int or List[int]
                                           num_layers=2,
                                           bidirectional= False,
                                           dilation=1, stride=1, dropout=0.5)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.lstm4 = cf.Conv3dPeepholeLSTM(self.nb_filter[2],  # Corresponds to input size
                                           out_channels= self.nb_filter[3] ,  # Corresponds to hidden size
                                           kernel_size=(3, 3, 3),  # Int or List[int]
                                           num_layers=2,
                                           bidirectional= False,
                                           dilation=1, stride=1, dropout=0.5)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # self.fc5 = nn.Linear(self.s_sz*1*2*1*nb_filter[3]*2, 128)
        self.fc5 = nn.Linear(1 * 2 * 1 * nb_filter[3], 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.fc7 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax()

    def forward(self, x): # ((time * batch_sz) *  30 * 36 * 30)
        x = x.view(self.s_sz,self.b_sz, 1, 30, 36, 30) # (time * batch_sz * 1 * 30 * 36 * 30)

        # https://pytorch.org/docs/stable/nn.html?highlight=rnn#torch.nn.RNN
        x, _ = self.lstm1(x) # (time * batch_sz * 1 * 30 * 36 * 30)
        x = x.view(self.s_sz*self.b_sz, self.nb_filter[0], 30, 36, 30)  # ((time * batch_sz) * self.nb_filter[0] * 30 * 36 * 30)
        x = self.pool1(x) # ((time * batch_sz) * 1 * 15 * 18 * 15)

        x = x.view(self.s_sz, self.b_sz, self.nb_filter[0], 15, 18, 15) # (time * batch_sz * self.nb_filter[0] * 15 * 18 * 15)
        x, _ = self.lstm2(x) # (time * batch_sz * self.nb_filter[0] * 15 * 18 * 15)
        x = x.contiguous().view(self.s_sz* self.b_sz, self.nb_filter[1], 15, 18, 15)  # ((time * batch_sz) * self.nb_filter[1] * 15 * 18 * 15)
        x = self.pool2(x) # ((time * batch_sz) * 1 * 7 * 9 * 7)

        x = x.view(self.s_sz, self.b_sz, self.nb_filter[1], 7, 9, 7) # (time * batch_sz * self.nb_filter[1] * 7 * 9 * 7)
        x, _ = self.lstm3(x) # (time * batch_sz * self.nb_filter[1] * 7 * 9 * 7)
        x = x.contiguous().view(self.s_sz*self.b_sz, self.nb_filter[2], 7, 9, 7)  # ((time * batch_sz) * self.nb_filter[2] * 7 * 9 * 7)
        x = self.pool3(x) # ((time * batch_sz) * 1 * 3 * 4 * 3)

        x = x.view(self.s_sz, self.b_sz, self.nb_filter[2], 3, 4, 3) # (time * batch_sz * self.nb_filter[2] *  3 * 4 * 3)
        x, _ = self.lstm4(x) # (time * batch_sz * self.nb_filter[2] *  3 * 4 * 3)
        x = x.view(self.s_sz*self.b_sz, self.nb_filter[3], 3, 4, 3)  # ((time * batch_sz) * self.nb_filter[3] *  3 * 4 * 3)
        x = self.pool4(x) # ((time * batch_sz( * 1 * 1 * 2 * 1)

        x = x.view(self.s_sz, self.b_sz, self.nb_filter[3], 1, 2, 1) # (time * batch_sz * self.nb_filter[3] *  1 * 2 * 1)

        x = x[-1] # get the last output  batch_sz * self.nb_filter[3] *  1 * 2 * 1
        x = x.contiguous().view(self.b_sz, self.nb_filter[3] * 1 * 2 * 1)

        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        res = self.softmax(self.fc7(x))


        '''
        x = x.permute(1, 0, 2, 3, 4, 5) # (batch_sz * time * self.nb_filter[3] *  1 * 2 * 1
        x = x.contiguous().view(self.b_sz, self.s_sz*self.nb_filter[3]*1*2*1)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.relu(self.bn6(self.fc6(x)))
        x = self.dropout(x)
        res = self.softmax(self.fc7(x))
        '''


        return res


    '''
    x.shape
    torch.Size([20, 5, 1, 30, 36, 30])
    x.shape
    torch.Size([20, 5, 8, 30, 36, 30])
    x.shape
    torch.Size([100, 8, 30, 36, 30])
    x.shape
    torch.Size([100, 8, 15, 18, 15])
    x.shape
    torch.Size([20, 5, 8, 15, 18, 15])
    x.shape
    torch.Size([20, 5, 16, 15, 18, 15])
    x.shape
    torch.Size([100, 16, 15, 18, 15])
    x.shape
    torch.Size([100, 16, 7, 9, 7])
    x.shape
    torch.Size([20, 5, 16, 7, 9, 7])

    '''