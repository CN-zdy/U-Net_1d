import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1d, self).__init__()
        self.Conv1d_16 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            Swish(),
            nn.Conv1d(in_channels, out_channels, 16, 1, 7),
            nn.BatchNorm1d(out_channels),
            Swish(),
            nn.Conv1d(out_channels, out_channels, 16, 1, 7),
        )

        self.Conv1d_1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            Swish(),
            nn.Conv1d(in_channels, out_channels, 1, 1)
            )

    def forward(self, x):

        out = x

        out = self.Conv1d_16(out)
        pad = nn.ReplicationPad1d(padding=(1, 1))
        out = pad(out)
        x = self.Conv1d_1(x)
        out += x

        return out

class U_Net_1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_list, n_classes, n):
        super(U_Net_1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_list = filter_list
        self.n_classes = n_classes
        self.n = n

        self.maxpooling = nn.MaxPool1d(2)

        self.en_conv_1 = Conv1d(self.in_channels, self.filter_list[0])#, self.kernel_size, stride=2)
        self.en_conv_2 = Conv1d(self.filter_list[0], self.filter_list[1])#, self.kernel_size, stride=2)
        self.en_conv_3 = Conv1d(self.filter_list[1], self.filter_list[2])#, self.kernel_size, stride=2)
        self.en_conv_4 = Conv1d(self.filter_list[2], self.filter_list[3])#, self.kernel_size, stride=2)

        self.up_3 = nn.ConvTranspose1d(self.filter_list[3], self.filter_list[2], 2, 2)
        self.up_2 = nn.ConvTranspose1d(self.filter_list[2], self.filter_list[1], 2, 2)
        self.up_1 = nn.ConvTranspose1d(self.filter_list[1], self.filter_list[0], 2, 2)

        self.de_conv_3 = Conv1d(self.filter_list[3], self.filter_list[2])#, self.kernel_size, stride=2)
        self.de_conv_2 = Conv1d(self.filter_list[2], self.filter_list[1])#, self.kernel_size, stride=2)
        self.de_conv_1 = Conv1d(self.filter_list[1], self.filter_list[0])#, self.kernel_size, stride=2)

        self.Conv1d = nn.Sequential(
            nn.BatchNorm1d(self.filter_list[0]),
            Swish(),
            nn.Conv1d(self.filter_list[0], self.out_channels, 1, 1)
            # nn.ReLU(inplace=True),
        )

        # final prediction
        self.dense1 = nn.Linear(self.n, self.filter_list[0])
        self.dense2 = nn.Linear(self.filter_list[0], n_classes)

    def forward(self, x):

        x1 = self.en_conv_1(x)

        x2 = self.maxpooling(x1)
        x2 = self.en_conv_2(x2)

        x3 = self.maxpooling(x2)
        x3 = self.en_conv_3(x3)

        x4 = self.maxpooling(x3)
        x4 = self.en_conv_4(x4)

        d4 = self.up_3(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.de_conv_3(d4)

        d3 = self.up_2(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.de_conv_2(d3)

        d2 = self.up_1(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.de_conv_1(d2)

        d1 = self.Conv1d(d2)

        #d = d1.view(-1, 64 * 3 * 3)
        d = self.dense1(d1)
        d = d.mean(-2)
        d = self.dense2(d)

        return d