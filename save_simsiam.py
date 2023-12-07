import torch
from torchvision.models import resnet18
ckpt_path='simsiam-cifar10-252e1tvw-ep=999.ckpt'
model = resnet18()
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Identity()
print(f"load pretrained model from {ckpt_path}")
state = torch.load(ckpt_path,map_location='cpu')["state_dict"]
for k in list(state.keys()):
    if "encoder" in k:
        state[k.replace("encoder", "backbone")] = state[k]
    if "backbone" in k:
        state[k.replace("backbone.", "")] = state[k]
    del state[k]

model.load_state_dict(state, strict=False)


torch.save(model.state_dict(), "simsiam_weights.pth")