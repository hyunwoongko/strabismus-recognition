"""
@author : Hyunwoong
@when : 2020-03-15
@homepage : https://github.com/gusdnd852
"""
import os

import torch

from configuration import *
from model.cnn_models.resnet import Model
from utils.dataset_generator import DatasetGenerator

model_name = "2_0"
dir = os.listdir(root_path + "\\saved\\")
file_name = root_path + "\\saved\\model_" + model_name + ".pth"

model = Model().to(device)
model.load_state_dict(torch.load(file_name))
model.eval()

gen = DatasetGenerator(max_length=max_length, ratio=1.0, flatten=False, shuffle=False)
abnormal = gen.load_data(path=root_path + "\\data\\test\\abnormal\\", label=1)
abnormal = gen.make_tensor(abnormal)
ab_feature, ab_label, ab_name = abnormal

normal = gen.load_data(path=root_path + "\\data\\test\\normal\\", label=0, )
normal = gen.make_tensor(normal)
no_feature, no_label, no_name = normal

x = torch.cat([ab_feature, no_feature], dim=0).cuda()
y = torch.cat([ab_label, no_label], dim=0).cuda()
out = model(x)

_, idx = out.max(dim=1)
debug = out.detach().cpu().numpy()

acc = 0
names = ab_name + no_name

print("========================")
for idx, i in enumerate(zip(y, idx)):
    print(names[idx][0:3], " 예측 : ", i[1].item(), " 정답 : ", i[0].item())
    if i[0] == i[1]:
        acc += 1

print("acc : ", acc / (len(names)))
