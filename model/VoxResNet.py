import torch
import torch.nn as nn

class Res_Module(nn.Module):
    """for VoxresNet model using:
    BN-Rule-Conv,BN-Relu_Conv"""
    def __init__(self,in_ch,act_fn=nn.ReLU(inplace='True'),dropout=False):
        super(Res_Module,self).__init__()
        if dropout:
            self.conv=nn.Sequential(
                nn.BatchNorm3d(in_ch),
                act_fn,
                nn.Conv3d(in_ch, 64, kernel_size=3, stride=1, padding=1, ),
                nn.BatchNorm3d(64),
                act_fn,
                nn.Dropout(p=0.2),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.conv = nn.Sequential(
                nn.BatchNorm3d(in_ch),
                act_fn,
                nn.Conv3d(in_ch, 64, kernel_size=3, stride=1, padding=1, ),
                nn.BatchNorm3d(64),
                act_fn,
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            )

        self.reset_params()

    def forward(self, input):
        x=self.conv(input)
        x+=input
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                m.weight.data.normal(0,0.01)
                m.bias.data.zero_()


class BN_Act(nn.Module):
    def __init__(self,in_ch,act_fn=nn.ReLU(inplace=True)):
        super(BN_Act,self).__init__()
        self.conv=nn.Sequential(
            nn.BatchNorm3d(in_ch),
            act_fn,
        )

    def forward(self, input):
        return self.conv(input)


################## the main function ###########################
class VoxResNet_3D(nn.Module):
    def __init__(self,in_ch,out_ch,act_fn=nn.ReLU(inplace=True)):
        super(VoxResNet_3D,self).__init__()
        self.conv1=nn.Conv3d(in_ch,32,kernel_size=3,stride=1,padding=1)
        self.bn_act1=BN_Act(32,act_fn)

        self.conv2=nn.Conv3d(32,32,kernel_size=3,stride=1,padding=1)
        self.deconv1=nn.Conv3d(32,out_ch,kernel_size=3,stride=1,padding=1)
        self.clc1=nn.Conv3d(out_ch,out_ch,kernel_size=1,stride=1,padding=0)

        self.bn_act2 = BN_Act(32, act_fn)

        self.do1 = nn.Dropout(p=0.2)
        self.do2 = nn.Dropout(p=0.2)
        self.do3 = nn.Dropout(p=0.2)

        self.conv3=nn.Conv3d(32,64,kernel_size=3,stride=2,padding=1)

        self.res2=Res_Module(64,act_fn,dropout=False)
        self.res3=Res_Module(64,act_fn,dropout=False)
        self.deconv2=nn.ConvTranspose3d(64,out_ch,2,2,0)
        # nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.clc2=nn.Conv3d(out_ch,out_ch,1,1,0)

        self.bn_act3=BN_Act(64,act_fn)

        self.conv4=nn.Conv3d(64,64,3,2,1)

        self.res5=Res_Module(64,act_fn,dropout=False)
        self.res6=Res_Module(64,act_fn,dropout=False)
        self.deconv3=nn.ConvTranspose3d(64,out_ch,4,4,0)
        self.clc3=nn.Conv3d(out_ch,out_ch,1,1,0)

        self.bn_act4=BN_Act(64,act_fn)
        self.conv5=nn.Conv3d(64,64,3,2,1)

        self.res8=Res_Module(64,act_fn,dropout=False)
        self.res9=Res_Module(64,act_fn,dropout=False)

        self.deconv4=nn.ConvTranspose3d(64,out_ch,8,8,0)
        self.clc4=nn.Conv3d(out_ch,out_ch,1,1,0)

        self.reset_params()

    def forward(self, input):
        x1=input
        x1=self.conv1(x1)
        x1=self.bn_act1(x1)

        x1=self.conv2(x1)

        x1_out = self.deconv1(x1)
        x1_out=self.clc1(x1_out)

        x2=self.bn_act2(x1)

        x2=self.conv3(x2)
        x2=self.res2(x2)
        x2=self.res3(x2)

        x2_out = self.deconv2(x2)
        x2_out=self.clc2(x2_out)

        x3=self.bn_act3(x2)
        x3=self.conv4(x3)
        x3=self.res5(x3)
        x3=self.res6(x3)


        x3_out = self.deconv3(x3)
        x3_out=self.clc3(x3_out)

        x4=self.bn_act4(x3)
        x4=self.conv5(x4)
        x4=self.res8(x4)
        x4=self.res9(x4)

        x4_out=self.deconv4(x4)
        x4_out=self.clc4(x4_out)

        x_all=x1_out+x2_out+x3_out+x4_out

        return x_all


    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()

if __name__=="__main__":
    device=torch.device("cuda")
    img=torch.rand(1,1,40,40,40).to(device)

    model=VoxResNet_3D(1,2)
    model=model.to(device)

    # output=model(img)
    #
    # print(output.size())

    # print(model)

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l

    print("总参数数量和：" + str(k))






