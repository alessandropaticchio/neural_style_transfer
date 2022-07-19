import torchvision.models as models
import torch.nn


class VGG_NST(torch.nn.Module):
    def __init__(self):
        super(VGG_NST, self).__init__()

        self.conv_features = ["0", "5", "10", "19", "28"]
        self.mapping = {0: "conv1_1", 5: "conv2_1", 10: "conv3_1", 19: "conv_4_1", 28: "conv_5_1"}
        self.backbone = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = {}

        for l_num, l in enumerate(self.backbone):
            x = l(x)

            if str(l_num) in self.conv_features:
                features[self.mapping[l_num]] = x

        return features
