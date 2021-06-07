import torch
from torch import nn
import copy
from torch.nn import functional as F


class TemporalConvModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim):
        super(TemporalConvModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1) # b c t
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        # self.lstm = nn.LSTM()
        # self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature)
        # frames = self.conv_out(feature)
        return feature


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()  # default value is 0.5
        # self.bn = nn.BatchNorm1d(in_channels, eps=1e-08, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x, use_bn=False):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        # if use_bn:
        #     out = self.bn(out)
        # else:
        out = self.dropout(out)
        return x + out