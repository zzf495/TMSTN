import torch
import torch.nn as nn
from module.MicroCommunity import MicroCommunity
from module.SmoothAdaptiveSemanticsEmbedding import SmoothAdaptiveSemanticsEmbedding
import torch.nn.utils.weight_norm as weightNorm

class TMSTN(nn.Module):
    def __init__(self, backbone, num_classes=31, bottle_neck=True, dimension=512, gar=0.2):
        super(TMSTN, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.bottle_neck = bottle_neck
        self._out = 768 # dimension of vit
        self.bottle = feat_bottleneck(type='bn', feature_dim=self._out, bottleneck_dim=dimension)
        if self.bottle_neck:
            self.classifier = feat_classifier(type='wn', class_num=num_classes, bottleneck_dim=dimension)
            self.centers = MicroCommunity(num_classes=num_classes, dimension=dimension, gar=gar)
        else:
            self.classifier = feat_classifier(type='wn', class_num=num_classes, bottleneck_dim=self._out)
            self.centers = MicroCommunity(num_classes=num_classes, dimension=self._out, gar=gar)
        self.age_cs = SmoothAdaptiveSemanticsEmbedding(num_class=num_classes)

    def get_features(self, img):
        _, fea = self.backbone(img)
        fea = fea[:, 0]
        if self.bottle_neck:
            fea = self.bottle(fea)
        return fea

    def predict(self, img):
        fea = self.get_features(img)
        pred = self.classifier(fea)
        return pred, fea

    def get_sgd(self, lr, momentum, decay, is_source=False):
        params = [
            {'params': self.bottle.parameters(), 'lr': lr},
            {'params': self.age_cs.parameters(), 'lr': lr},
            {'params': self.backbone.parameters(), 'lr': lr / 10}
        ]
        if is_source:
            params.append({'params': self.classifier.parameters(), 'lr': lr})
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=decay)
        optimizer_centers = torch.optim.SGD([{'params': self.centers.parameters(), 'lr': lr}],
                                         lr=lr, momentum=momentum, weight_decay=decay)
        return optimizer, optimizer_centers

class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.type = type

    def zeros_weights(self):
        self.bottleneck.apply(init_weights)
        print('zeros the weight of feat_bottleneck.')

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            # self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)


    def zeros_weights(self):
        self.fc.apply(init_weights)
        print('zeros the weight of feat_classifier.')


    def forward(self, x):
        x = self.fc(x)
        return x

def init_weights(m):
    classname = m.__class__.__name__
    print(f"init_weights: {classname}")
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)