"""
modification made on the basis of link:https://github.com/liuyoude/STgram-MFN
"""
import math
from torch import nn
import torch
from losses import ArcMarginProduct


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


#https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Liu_8_t2.pdf
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]


class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(128, num_class)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)
        return out, feature


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for _ in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out
    

class TASTgramMFN(nn.Module):
    def __init__(self, num_classes, mode,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=True, m=0.7, s=30, sub=1
                 ):
        super().__init__()
        
        self.arcface = ArcMarginProduct(in_features=c_dim, out_features=num_classes,
                                        m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting)
        self.mode = mode
        
        if mode not in ['arcface', 'arcmix', 'noisy_arcmix']:
            raise ValueError('Choose one of [arcface, arcmix, noisy_arcmix]')
        
        self.temporal_attention = Temporal_Attention(feature_dim=c_dim)
        
    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label, train=True):
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        
        x_mel_temp_att = self.temporal_attention(x_mel).unsqueeze(1)
       
        x = torch.cat((x_t, x_mel, x_mel_temp_att), dim=1)
        
        out, feature = self.mobilefacenet(x)
        
        if self.mode == 'arcmix':
            if train:
                out = self.arcface(feature, label[0])
                out_shuffled = self.arcface(feature, label[1])
                return out, out_shuffled, feature
            else:
                out = self.arcface(feature, label)
                return out, feature
        
        else:
            out = self.arcface(feature, label)
            return out, feature
        
        
class Temporal_Attention(nn.Module):
  def __init__(self, feature_dim=128):
    super().__init__()
    
    self.feature_dim = feature_dim
    self.max_pool = nn.AdaptiveMaxPool1d(1)
    self.avg_pool = nn.AdaptiveAvgPool1d(1)
    self.sigmoid = nn.Sigmoid()
    
  def forward(self, x):
    # x: (B, 1, 128, 313)
    x = x.squeeze(1)
    
    x = x.transpose(1,2) # (B, 313, 128)

    x1 = self.max_pool(x) # (B, 313, 1)
    x2 = self.avg_pool(x) # (B, 313, 1)
    
    feats = x1 + x2
    
    feats = feats.repeat(1, 1, self.feature_dim)
    
    refined_feats = self.sigmoid(feats).transpose(1,2) * x.transpose(1,2)
    return refined_feats