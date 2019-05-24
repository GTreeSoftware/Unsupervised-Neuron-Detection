"""This is my model of UNet 3D"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class conv3d(nn.Module):
    """conv---BN----Relu"""
    def __init__(self,in_ch,out_ch,ker_size=3,BN=True,act_fn=nn.ReLU(inplace=True)):
        super(conv3d,self).__init__()
        if BN:
            self.conv=nn.Sequential(
                nn.Conv3d(in_ch,out_ch,kernel_size=ker_size,stride=1,padding=1),
                nn.BatchNorm3d(out_ch),
                act_fn,
            )
        else:
            self.conv=nn.Sequential(
                nn.Conv3d(in_ch,out_ch,kernel_size=ker_size,stride=1,padding=1),
                act_fn,
            )

    def forward(self, input):
        return self.conv(input)

class double_conv3d(nn.Module):
    """conv3d--conv3d"""
    def __init__(self,in_ch,out_ch,ker_size=3,BN=True,actfun=nn.ReLU(inplace=True)):
        super(double_conv3d,self).__init__()
        self.conv=nn.Sequential(
            conv3d(in_ch,out_ch,ker_size,BN,actfun),
            conv3d(out_ch, out_ch, ker_size, BN, actfun),
        )

    def forward(self, input):
        return self.conv(input)


class inconv(nn.Module):
    """The input: double_conv3d"""
    def __init__(self,in_ch,out_ch,ker_size=3,BN=True,actfun=nn.ReLU(inplace=True)):
        super(inconv,self).__init__()
        self.conv=double_conv3d(in_ch,out_ch,ker_size,BN,actfun)

    def forward(self, input):
        return self.conv(input)


class down(nn.Module):
    """Max---double_conv3d"""
    def __init__(self,in_ch,out_ch,ker_size=3,BN=True,actfun=nn.ReLU(inplace=True)):
        super(down,self).__init__()
        self.conv=nn.Sequential(
            nn.MaxPool3d(kernel_size=2,stride=2,padding=0),
            double_conv3d(in_ch,out_ch,ker_size,BN,actfun),
        )

    def forward(self, input):
        return self.conv(input)


class up(nn.Module):
    """convup---double_conv2d"""
    def __init__(self,in_ch,out_ch,ker_size=3,BN=True,actfun=nn.ReLU(inplace=True),bilinear=False):
        super(up,self).__init__()

        if bilinear:
            self.conv_up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        else:
            self.conv_up=nn.ConvTranspose3d(in_ch,out_ch,kernel_size=3, stride=2, padding=1,output_padding=1)

        if BN:
            self.conv_up=nn.Sequential(
                self.conv_up,
                nn.BatchNorm3d(out_ch),
                # whether act_fn is needed??
            )

        self.conv=double_conv3d(in_ch,out_ch,ker_size=3,BN=True,actfun=nn.ReLU(inplace=True))

    def forward(self, input1,input2):
        x1=self.conv_up(input1)
        # the position of cat
        x2=torch.cat([input2,x1],dim=1)
        return self.conv(x2)

class out_conv(nn.Module):
    def __init__(self,in_ch,out_ch,ker_size=1):
        super(out_conv,self).__init__()
        self.conv=nn.Conv3d(in_ch,out_ch,kernel_size=ker_size)

    def forward(self, input):
        return self.conv(input)


class UNet_3D(nn.Module):
    def __init__(self,in_ch,out_ch,st_filter=16):
        super(UNet_3D,self).__init__()
        self.inc=inconv(in_ch,st_filter)
        self.down1=down(st_filter,st_filter*2)
        self.down2=down(st_filter*2,st_filter*4)
        self.down3=down(st_filter*4,st_filter*8)
        self.down4=down(st_filter*8,st_filter*16)

        self.up1=up(st_filter*16,st_filter*8)
        self.up2=up(st_filter*8,st_filter*4)
        self.up3=up(st_filter*4,st_filter*2)
        self.up4=up(st_filter*2,st_filter*1)

        self.outc=out_conv(st_filter,out_ch)

    def forward(self, input):
        x1=self.inc(input)
        xd_1=self.down1(x1)
        xd_2=self.down2(xd_1)
        xd_3=self.down3(xd_2)
        xd_4=self.down4(xd_3)

        xu_1=self.up1(xd_4,xd_3)
        xu_2=self.up2(xu_1,xd_2)
        xu_3=self.up3(xu_2,xd_1)
        xu_4=self.up4(xu_3,x1)

        xout=self.outc(xu_4)

        return xout


if __name__=="__main__":
    model1=UNet_3D(1,2)
    print(model1)


















if __name__=="__main__":
    model1=conv3d(1,2)
    print(model1)


