from torch import nn
import torch
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))

class SegLieaner(nn.Module):
    def __init__(self,in_channels,classes=40):
        super(SegLieaner, self).__init__()
        self.last_layer=nn.Linear(in_channels, classes)
        self._GCONST_= -0.9189385332046727
    def forward(self,z, log_jac_det):
        out=(256.0 * self._GCONST_ - 0.5 * z ** 2 + torch.unsqueeze(log_jac_det, dim=1))/256.0
        out=self.last_layer(out)
        return out
class flow_seg_decoder(nn.Module):
    def __init__(self, args,in_channels, classes=40):
        super(flow_seg_decoder, self).__init__()
        self.flow_decoder=flow_model(args,in_channels)
        self.last_layer=SegLieaner(in_channels,classes)
    def forward(self,x):
        z, log_jac_det=self.flow_decoder(x)
        out=self.last_layer(z, log_jac_det)
        return out



def flow_model(args, in_channels):
    args.coupling_layers=8
    args.clamp_alpha=1.9
    coder = Ff.SequenceINN(in_channels)
    print('Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.coupling_layers):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def conditional_flow_model(args, in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Conditional Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(8):  # args.coupling_layers=8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(128,), subnet_constructor=subnet_fc, affine_clamping=1.9,
            global_affine_type='SOFTPLUS', permute_soft=True)#args.pos_embed_dim=128 args.clamp_alpha=1.9
    return coder