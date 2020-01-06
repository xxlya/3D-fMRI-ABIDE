import torch
import model.cnn_rnn.convolutional_rnn as cf
from torch.nn.utils.rnn import pack_padded_sequence

net = cf.Conv3dGRU(in_channels=1,  # Corresponds to input size
                                  out_channels=5,  # Corresponds to hidden size
                                  kernel_size=(3,3,3),  # Int or List[int]
                                  num_layers=3,
                                  bidirectional=False,
                                  dilation=1, stride=3, dropout=0.5)
#x = pack_padded_sequence(torch.randn(20, 2, 1, 7, 9, 7),[20]*2)
x = torch.randn(20, 2, 1, 7, 9, 7)
print(net)
y, h = net(x)
print(y.data.shape)
print(h.shape)

x = pack_padded_sequence(torch.randn(20, 2, 1, 7, 9, 7),[20]*2)
y, h = net(x)
print(y.data.shape)
print(h.shape)

cell = cf.Conv2dLSTMCell(in_channels=3, out_channels=5, kernel_size=3).cuda()
time = 6
input = torch.randn(time, 16, 3, 10, 10).cuda()
output = []
for i in range(time):
    if i == 0:
        hx, cx = cell(input[i])
    else:
        hx, cx = cell(input[i], (hx, cx))
    output.append(hx)
