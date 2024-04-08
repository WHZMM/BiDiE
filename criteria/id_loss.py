# https://github.com/TreB1eN/InsightFace_Pytorch
# """
# ir_se55 on 112x112 face landmark avg:
# [[38.29459953 51.69630051]
# [73.53179932 51.50139999]
# [56.02519989 71.73660278]
# [41.54930115 92.3655014 ]
# [70.72990036 92.20410156]]
# """
# EG3D(256x256)landmark:
# 95, 99; 162, 99; 128, 141; 94, 168; 161, 168
# x = x[:, :, 26:228, 5:207]



import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = x[:, :, 26:228, 5:207]  # crop 256 img for EG3D
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y) 
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count, id_logs


if __name__ == "__main__":
    print("这个是criteria/id_loss.py里面的测试代码,训练时候不应该执行!!!")
    device = "cuda:0"
    id_loss = IDLoss().to(device).eval()
    print("测试完成")

