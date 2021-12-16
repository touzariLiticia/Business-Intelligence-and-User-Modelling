import torch
import torch.nn as nn
from torchvision import models


def get_ftr_extractor(pretrained=True, model_name="densenet201"):
    if model_name == "densenet201":
        ftrs = models.densenet201(pretrained=True).features
        n = 1920
    elif model_name == "densenet121":
        ftrs = models.densenet121(pretrained=True).features
        n = 1024
    elif model_name == "mobilenet_v2":
        ftrs = models.mobilenet_v2(pretrained=True).features
        n = 1280
    else:
        m = models.shufflenet_v2_x1_0(pretrained=False)
        ftrs = nn.Sequential(*list(m.children())[:-2])
        n = 464
    return ftrs, n


class Net(nn.Module):
    def __init__(
        self,
        backbone_name,
        num_classes,
        num_outgame_subclasses,
        dropout_fts=0.1,
        pretrained=True,
    ):
        super(Net, self).__init__()
        self.dropout_fts = dropout_fts
        self.ftrs, self.num_fts = get_ftr_extractor(
            pretrained=pretrained, model_name=backbone_name
        )
        self.num_classes = num_classes
        self.drop = nn.Dropout(dropout_fts)
        self.fc_ingame = nn.Linear(self.num_fts, 2)
        self.fc_game = nn.Linear(self.num_fts, num_classes)
        self.fc_outgame_subclasses = nn.Linear(
            self.num_fts, num_outgame_subclasses
        )
        self.fc_distill_projection = nn.Linear(self.num_fts, self.num_fts)

    def forward(self, x):
        x_ftrs = self.ftrs(x).mean(3).mean(2)
        x_ftrs = self.drop(x_ftrs)
        ftrs_for_distill = self.fc_distill_projection(x_ftrs)
        x_ingame = self.fc_ingame(x_ftrs)
        x_game = self.fc_game(x_ftrs)
        x_outgame_subclasses = self.fc_outgame_subclasses(x_ftrs)
        return x_ingame, x_game, x_outgame_subclasses, ftrs_for_distill, x_ftrs
