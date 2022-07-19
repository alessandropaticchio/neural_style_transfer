from torchvision.transforms import transforms


def compute_G_style(style_features):
    G_styles = {}
    for k, style_feat in style_features.items():
        _, channel, height, width = style_feat.shape

        G_style = style_feat.view(channel, height * width).mm(style_feat.view(channel, height * width).t())
        G_styles[k] = G_style

    return G_styles


def get_preprocessing_transform(image_size):
    img_preprocess = transforms.Compose(
        [transforms.Resize((image_size, image_size)),
         transforms.ToTensor()]
    )
    return img_preprocess
