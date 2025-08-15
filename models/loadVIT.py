from numpy import load
from models.modeling import VisionTransformer, CONFIGS

# https://console.cloud.google.com/storage/browser/vit_models?inv=1&invt=AbyfNA

def load_vit_model(num_class=1000):
    img_size = [224, 224]
    model_type = 'ViT-B_16'
    path = '../pretrained_models/imagenet21k+imagenet2012-FViT-B_16-224.npz'
    print(f'load vit from {path}')
    path = load(path)
    config = CONFIGS[model_type]
    model = VisionTransformer(config, img_size, num_classes=num_class, zero_head=True)
    model.load_from(path)
    return model

def load_vit_model_resnet(num_class=1000):
    img_size = [224, 224]
    model_type = 'R50-ViT-B_16'
    path = '../pretrained_models/imagenet21k+imagenet2012_R50+ViT-B_16.npz'
    print(f'load vit from {path}')
    path = load(path)
    config = CONFIGS[model_type]
    model = VisionTransformer(config, img_size, num_classes=num_class, zero_head=True)
    model.load_from(path)
    return model
