import os

import torch
import torch.nn as nn
import einops
import copy
# import ipdb
from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
import mmaction
from mmaction.models import build_model
from mmcv import Config
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer, BertAttention, BertIntermediate, BertOutput
from .temporalConv import TemporalConvModel


class audnet(nn.Module):
    def __init__(self):
        super(audnet, self).__init__()
        self.conv1 = nn.Conv1d(80, 128, kernel_size=7, padding=3, stride=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))

    def forward(self, x):  # [bs,128,80] -> [bs, 80, 128]
        x = einops.rearrange(x, 'b 1 t c -> b c t')
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool(x)
        x = F.relu(self.conv4(x))
        x = self.avg_pool(x).flatten(1)

        return x

class CSN(nn.Module):
    def __init__(self):
        super().__init__()
        mmaction_root = os.path.dirname(os.path.abspath(mmaction.__file__))
        config_file = os.path.join(mmaction_root, os.pardir, 'configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py')
        cfg = Config.fromfile(config_file)

        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        state_dict = torch.load('../ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth')
        print('load from ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth', flush=True)
        model.load_state_dict(state_dict['state_dict'])
        del model.cls_head
        self.model = model

    @staticmethod
    def forward_train(model, imgs, labels=None):
        """Defines the computation performed at every call when training."""
        losses = dict()

        x = model.extract_feat(imgs)
        if model.with_neck:
            x, loss_aux = model.neck(x, labels.squeeze())
            losses.update(loss_aux)

        return x

    def forward(self, x):
        """(B, C, T, H, W)"""
        x = self.model.extract_feat(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, seq_length):
        super().__init__()
        config = BertConfig(
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=8,
            intermediate_size=2048,
        )
        # self.self0 = BertAttention(config)
        # self.self1 = BertAttention(config)
        # self.cross = BertAttention(config)
        #
        # self.intermediate0 = BertIntermediate(config)
        # self.output0 = BertOutput(config)
        #
        # self.intermediate1 = BertIntermediate(config)
        # self.output1 = BertOutput(config)
        self.seq_length = seq_length
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(4)])
        self.conv1 = nn.Conv2d(1, 512, kernel_size=(seq_length, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(512, 1, 1))
        print('BertLayer')

    def forward(self, x):
        """(B, T, C)"""
        for layer in self.layers:
            x = layer(x)[0]
        x = x.unsqueeze(1)
        # b 1 t c
        x = self.conv1(x)  # (16, 512, 1, 512)
        x = self.max3d(x).flatten(1)
        return x


class mc3_18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.video.mc3_18(pretrained=True)
        self.in_features = self.model.fc.in_features
        del self.model.fc
        del self.model.avgpool

    def forward(self, x):
        """(B, 3, T, H, W)"""

        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # (B, C, T, H, W)

        return x


class temporalNet(nn.Module):
    def __init__(self,t_conv_layers=12, t_conv_hidden_size=256, input_dim=1024, lstm_layers=1, lstm_hidden_size=256, seq_length=32, channel=512):
        super(temporalNet, self).__init__()
        self.channel = channel
        self.temporal_conv = TemporalConvModel(t_conv_layers, t_conv_hidden_size, input_dim)
        self.lstm = nn.LSTM(input_size=t_conv_hidden_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(seq_length, 1))

    def forward(self, x, hx=None):
        # b c t
        x = self.temporal_conv(x)

        x = x.permute(0, 2, 1).contiguous()
        x, (_, _) = self.lstm(x, hx)
        x = x.unsqueeze(1)
        # b 1 t c
        x = self.conv1(x)
        return x


class GEBDHead(nn.Module):
    def __init__(self, frames_per_side, use_self_att):
        super().__init__()
        self.use_self_att = use_self_att
        if self.use_self_att:
            self.att = AttentionLayer(frames_per_side * 2)
        in_feat_dim = 512
        num_classes = 2
        channel = 512
        self.temporal_model = temporalNet(t_conv_layers=4, input_dim=in_feat_dim, seq_length=frames_per_side*2)
        self.aud_net = audnet()
        self.max3d = nn.MaxPool3d(kernel_size=(channel, 1, 1))
        self.project_global = nn.Linear(2048, 256)

        self.fc = nn.Linear(
            in_features=2 * self.temporal_model.lstm.hidden_size + (512 if self.use_self_att else 0) + 512,
            out_features=num_classes,
            bias=True
        )

    def forward(self, x_context, aud_feat, global_feats):
        """
        x_context: (B, C, T), global_feats: (B, 2048)
        """
        B = x_context.shape[0]
        global_feats = self.project_global(global_feats)  # (16, 256)
        hx = global_feats.unsqueeze(0).repeat(2, 1, 1)
        hx = (hx, hx)

        att_context = None
        if self.use_self_att:
            att_context = self.att(einops.rearrange(x_context, 'b c t -> b t c'))

        x_context = self.temporal_model(x_context, hx)
        x_context = self.max3d(x_context).view(B, -1)  # b 1 1 c
        aud_feat = self.aud_net(aud_feat)

        if att_context is not None:
            logits = self.fc(torch.cat([x_context, att_context, aud_feat], dim=1))
        else:
            logits = self.fc(x_context)

        return logits


class mc3lstmaudioGEBDCascade(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, frames_per_side=5, use_flow=False,
                 use_backbone_flow=False, lstm_hidden_size=256, train_multi_head=True, filter_thresh=[0.2, 0.3]):
        super(mc3lstmaudioGEBDCascade, self).__init__()
        self.use_flow = use_flow
        self.num_classes = num_classes
        self.frames_per_side = frames_per_side
        self.channel = 512
        # self.backbone_name = backbone
        self.lstm_hidden_size = lstm_hidden_size
        # self.backbone = mc3_18(pretrained=pretrained)
        self.backbone = CSN()
        if use_flow:
            self.backbone_flow = mc3_18(pretrained=pretrained)
            # self.backbone_flow = CSN()
            self.flow_proj = nn.Sequential(nn.Conv3d(2, 3, kernel_size=1), nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool3d((frames_per_side * 2, 1, 1))
        self.project_rgb_and_flow = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512)
        )

        self.use_self_att = True
        self.heads = nn.ModuleList([
            GEBDHead(frames_per_side=frames_per_side, use_self_att=self.use_self_att) for _ in range(3)
        ])
        # self.head = GEBDHead(frames_per_side=frames_per_side, use_self_att=self.use_self_att)
        self.train_multi_head = train_multi_head

        s = 'csn + dynamic + global + cascade'
        if self.use_self_att:
            s += ' + self att'
        if use_flow:
            s += ' + flow'
        self.filter_thresh = filter_thresh
        print(s, flush=True)
        self.thresholds = filter_thresh

    def forward(self, inputs, targets=None):
        if self.training:
            assert targets is not None, 'Training need targets!!'
        x_rgb = inputs['inp']
        aud_feat = inputs['aud_feats']
        # global_img = inputs['global_img']
        # global_img = einops.rearrange(global_img, 'b t c h w -> b c t h w')
        # global_feats = self.backbone_global(global_img)
        # global_feats = F.adaptive_avg_pool3d(global_feats, 1).flatten(1).detach()
        # global_feats = self.project_global(global_feats)  #  (16, 256)
        global_feats = inputs['global_feats']

        x_rgb = einops.rearrange(x_rgb, 'b t c h w -> b c t h w')
        x_rgb = self.backbone(x_rgb)

        if self.use_flow:
            x_flow = inputs['flow']
            x_flow = einops.rearrange(x_flow, 'b t c h w -> b c t h w')
            x_flow = self.backbone_flow(self.flow_proj(x_flow))

            x_rgb = self.avg_pool(x_rgb).flatten(2)
            x_flow = self.avg_pool(x_flow).flatten(2)

            x_context = torch.cat([x_rgb, x_flow], dim=1)
        else:
            x_context = self.avg_pool(x_rgb).flatten(2)
        # b c t h w
        # print(x.shape)
        # x_context = self.avg_pool(x).flatten(2)  # (16, 1024, 16)
        # x_context = einops.rearrange(x, 'b c t 1 1-> b c t', b=B)
        x_context = self.project_rgb_and_flow(einops.rearrange(x_context, 'b c t -> b t c'))
        x_context = einops.rearrange(x_context, 'b t c -> b c t')

        # logits = self.head(x_context, global_feats)

        # thresholds = [0.2, 0.3]
        targets_list = []
        if self.training:
            targets_list = list(torch.unbind(targets, dim=1))

        # total_loss = 0
        losses = {}
        logits_list = []
        logits = self.heads[0](x_context, aud_feat, global_feats)
        logits_list.append(logits)
        if self.training:
            loss = F.cross_entropy(logits, targets_list[0])
            # total_loss += loss
            # losses.append(loss)
            losses['stage1'] = loss

        # batch_idx = torch.arange(x_rgb.shape[0], dtype=torch.int64, device=x_rgb.device)
        ignore_index = -100
        for stage, head in enumerate(self.heads[1:]):
            positive_probs = F.softmax(logits.detach(), dim=1)[:, 1]
            logits = head(x_context, aud_feat, global_feats)
            if self.training:
                t = targets_list[stage + 1].clone()
                with torch.no_grad():
                    t[positive_probs < self.thresholds[stage]] = ignore_index
                loss = F.cross_entropy(logits, t, ignore_index=ignore_index)
                # total_loss += loss
                # losses.append(loss)
                losses[f'staget{stage+1}'] = loss
            logits_list.append(logits)

        if self.training:
            return losses
        logits = sum(logits_list) / len(logits_list)

        return logits

    def fix_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


if __name__ == '__main__':
    net = mc3lstmGEBD(use_flow=True, frames_per_side=16)
    inp = torch.randn(2, 32, 3, 224, 224)
    flow = torch.randn(2, 32, 2, 224, 224)
    print(net([inp, flow]).shape)