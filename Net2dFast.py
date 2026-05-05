import torch
from typing import NamedTuple

class ModelOutput(NamedTuple):
    """Output of the detection model.
    Each of the tensors has a shape of `(batch_size, num_channels, spec_height, spec_width)`.
    Where `spec_height` and `spec_width` are the height and width of the input spectrograms."""

    pred_det: torch.Tensor #Tensor with predict detection probabilities.
    pred_size: torch.Tensor #Tensor with predicted bounding box sizes
    pred_class: torch.Tensor #Tensor with predicted class probabilities.
    pred_class_un_norm: torch.Tensor # Tensor with predicted class probabilities before softmax.
    features: torch.Tensor # Tensor with intermediate features.

class SelfAttention(torch.nn.Module):
    def __init__(self, ip_dim, att_dim):
        super(SelfAttention, self).__init__()
        # Note, does not encode position information (absolute or realtive)
        self.temperature = 1.0
        self.att_dim = att_dim
        self.key_fun = torch.nn.Linear(ip_dim, att_dim)
        self.val_fun = torch.nn.Linear(ip_dim, att_dim)
        self.que_fun = torch.nn.Linear(ip_dim, att_dim)
        self.pro_fun = torch.nn.Linear(att_dim, ip_dim)

    def forward(self, x):
        x = x.squeeze(2).permute(0, 2, 1)
        kk = torch.matmul(x, self.key_fun.weight.T) + self.key_fun.bias.unsqueeze(0).unsqueeze(0)
        qq = torch.matmul(x, self.que_fun.weight.T) + self.que_fun.bias.unsqueeze(0).unsqueeze(0)
        vv = torch.matmul(x, self.val_fun.weight.T) + self.val_fun.bias.unsqueeze(0).unsqueeze(0)
        kk_qq = torch.bmm(kk, qq.permute(0, 2, 1)) / (self.temperature * self.att_dim)
        att_weights = torch.nn.functional.softmax(kk_qq, 1)  # each col of each attention matrix sums to 1
        att = torch.bmm(vv.permute(0, 2, 1), att_weights)
        op = torch.matmul(att.permute(0, 2, 1), self.pro_fun.weight.T) + self.pro_fun.bias.unsqueeze(0).unsqueeze(0)
        op = op.permute(0, 2, 1).unsqueeze(2)
        return op
        
class ConvBlockDownCoordF(torch.nn.Module):
    def __init__( self, in_chn, out_chn, ip_height, k_size=3, pad_size=1, stride=1):
        super(ConvBlockDownCoordF, self).__init__()
        self.coords = torch.nn.Parameter(torch.linspace(-1, 1, ip_height)[None, None, ..., None], requires_grad=False) # linspace start end step
        self.conv = torch.nn.Conv2d(in_chn + 1, out_chn, kernel_size=k_size, padding=pad_size, stride=stride,) # 2D convolution 
        self.conv_bn = torch.nn.BatchNorm2d(out_chn) # Batch Normalization

    def forward(self, x):
        freq_info = self.coords.repeat(x.shape[0], 1, 1, x.shape[3]) 
        x = torch.cat((x, freq_info), 1) # concatenates freq_info ############
        x = torch.nn.functional.max_pool2d(self.conv(x), 2, 2)
        x = torch.nn.functional.relu(self.conv_bn(x), inplace=True) # Rectified Linear Unit, it outputs the input directly if it is positive, and zero otherwise
        return x
        
class ConvBlockUpF(torch.nn.Module):
    def __init__(self, in_chn, out_chn, ip_height, k_size=3, pad_size=1, up_mode="bilinear", up_scale=(2, 2)):
        super(ConvBlockUpF, self).__init__()
        self.up_scale = up_scale
        self.up_mode = up_mode
        self.coords = torch.nn.Parameter(torch.linspace(-1, 1, ip_height * up_scale[0])[None, None, ..., None],requires_grad=False,)
        self.conv = torch.nn.Conv2d(in_chn + 1, out_chn, kernel_size=k_size, padding=pad_size)
        self.conv_bn = torch.nn.BatchNorm2d(out_chn)

    def forward(self, x):
        op = torch.nn.functional.interpolate(x,size=(x.shape[-2] * self.up_scale[0], x.shape[-1] * self.up_scale[1]),mode=self.up_mode,align_corners=False)
        freq_info = self.coords.repeat(op.shape[0], 1, 1, op.shape[3]) 
        op = torch.cat((op, freq_info), 1)  # concatenates freq_info #############
        op = self.conv(op)
        op = torch.nn.functional.relu(self.conv_bn(op), inplace=True)
        return op

class Net2dFast(torch.nn.Module):
    def __init__(self, num_filts, num_classes=0, ip_height=128):
        super().__init__()
        print(f"Net2dFast {num_filts=} {num_classes=} {ip_height=}")
        self.num_classes = num_classes
        self.num_filts = num_filts
        self.ip_height_rs = ip_height
        self.bneck_height = self.ip_height_rs // 32

        # encoder
        self.conv_dn_0 = ConvBlockDownCoordF(1, num_filts // 4, self.ip_height_rs, k_size=3, pad_size=1, stride=1)
        self.conv_dn_1 = ConvBlockDownCoordF(num_filts // 4, num_filts // 2, self.ip_height_rs // 2, k_size=3, pad_size=1, stride=1)
        self.conv_dn_2 = ConvBlockDownCoordF(num_filts // 2, num_filts, self.ip_height_rs // 4,  k_size=3,pad_size=1,stride=1)
        
        self.conv_dn_3 = torch.nn.Conv2d(num_filts, num_filts * 2, 3, padding=1)
        self.conv_dn_3_bn = torch.nn.BatchNorm2d(num_filts * 2)

        # bottleneck
        self.conv_1d = torch.nn.Conv2d(num_filts * 2, num_filts * 2, (self.ip_height_rs // 8, 1), padding=0)
        self.conv_1d_bn = torch.nn.BatchNorm2d(num_filts * 2)
        
        self.att = SelfAttention(num_filts * 2, num_filts * 2)

        # decoder
        self.conv_up_2 = ConvBlockUpF(num_filts * 2, num_filts // 2, self.ip_height_rs // 8)
        self.conv_up_3 = ConvBlockUpF(num_filts // 2, num_filts // 4, self.ip_height_rs // 4)
        self.conv_up_4 = ConvBlockUpF(num_filts // 4, num_filts // 4, self.ip_height_rs // 2)

        # output +1 to include background class for class output
        self.conv_op = torch.nn.Conv2d(num_filts // 4, num_filts // 4, kernel_size=3, padding=1)
        self.conv_op_bn = torch.nn.BatchNorm2d(num_filts // 4)
        self.conv_size_op = torch.nn.Conv2d(num_filts // 4, 2, kernel_size=1, padding=0)
        self.conv_classes_op = torch.nn.Conv2d(num_filts // 4, self.num_classes + 1, kernel_size=1, padding=0)
            
    def forward(self, ip, return_feats=False) -> ModelOutput:
        # encoder
        x1 = self.conv_dn_0(ip)
        x2 = self.conv_dn_1(x1)
        x3 = self.conv_dn_2(x2)
        x3 = torch.nn.functional.relu(self.conv_dn_3_bn(self.conv_dn_3(x3)), inplace=True)
        # bottleneck
        x = torch.nn.functional.relu(self.conv_1d_bn(self.conv_1d(x3)), inplace=True)
        x = self.att(x)
        x = x.repeat([1, 1, self.bneck_height * 4, 1])
        # decoder
        x = self.conv_up_2(x + x3)
        x = self.conv_up_3(x + x2)
        x = self.conv_up_4(x + x1)
        # output
        x = torch.nn.functional.relu(self.conv_op_bn(self.conv_op(x)), inplace=True)
        cls = self.conv_classes_op(x)
        comb = torch.softmax(cls, 1)
        return ModelOutput(pred_det=comb[:, :-1, :, :].sum(1).unsqueeze(1), pred_size=torch.nn.functional.relu(self.conv_size_op(x), inplace=True),
            pred_class=comb, pred_class_un_norm=cls, features=x)
