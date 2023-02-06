from torchvision.models import resnet18
from torchvision.models import resnet50
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim


def unified_net():
    u_net = resnet50(pretrained=True)
    u_net.conv1 = nn.Identity()
    u_net.bn1 = nn.Identity()
    u_net.relu = nn.Identity()
    u_net.maxpool = nn.Identity()
    return u_net


def unified_net_l1():
    u_net = resnet50(pretrained=True)
    u_net.conv1 = nn.Identity()
    u_net.bn1 = nn.Identity()
    u_net.relu = nn.Identity()
    u_net.maxpool = nn.Identity()
    u_net.layer1 = nn.Identity()
    return u_net


class MultiScaleNet(nn.Module):
    def __init__(self):
        super(MultiScaleNet, self).__init__()
        self.large_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.mid_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.small_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.unified_net = unified_net()

    def forward(self, x):
        [small_imgs, mid_imgs, large_imgs] = x
        z1 = self.small_net(small_imgs)
        z2 = self.mid_net(mid_imgs)
        z3 = self.large_net(large_imgs)
        y1 = self.unified_net(z1)
        y2 = self.unified_net(z2)
        y3 = self.unified_net(z3)
        return z1, z2, z3, y1, y2, y3

    def small_forward(self, small_imgs):
        x = self.small_net(small_imgs)
        x = self.unified_net(x)
        return x

    def mid_forward(self, mid_imgs):
        x = self.mid_net(mid_imgs)
        x = self.unified_net(x)
        return x

    def large_forward(self, large_imgs):
        x = self.large_net(large_imgs)
        x = self.unified_net(x)
        return x


class SupervisedMultiScaleNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.multi_scale_net = MultiScaleNet()

    def training_step(self, batch, batch_idx):
        x, y = batch
        [small_imgs, mid_imgs, large_imgs] = x
        z1, z2, z3, y1, y2, y3 = self.multi_scale_net(small_imgs, mid_imgs, large_imgs)
        mse_loss1 = nn.functional.mse_loss(y1, y)
        mse_loss2 = nn.functional.mse_loss(y2, y)
        mse_loss3 = nn.functional.mse_loss(y3, y)
        total_loss = mse_loss1 + mse_loss2 + mse_loss3
        loss_dict = {
            "mse_loss1": mse_loss1,
            "mse_loss2": mse_loss2,
            "mse_loss3": mse_loss3,
            "total_loss": total_loss}
        self.log_dict(loss_dict)

        return total_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    small_imgs = torch.rand(8, 3, 32, 32)
    mid_imgs = torch.rand(8, 3, 128, 128)
    large_imgs = torch.rand(8, 3, 224, 224)
    m_model = MultiScaleNet()
    input = [small_imgs, mid_imgs, large_imgs]
    out = m_model(input)
    print("end")
