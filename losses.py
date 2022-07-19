import torch


def compute_content_loss(content_features, generated_features):
    content_loss = 0.
    for gen_feat, content_feat in zip(content_features.values(), generated_features.values()):
        content_loss += torch.mean((gen_feat - content_feat).pow(2))
    return content_loss


def compute_style_loss(style_features, generated_features, G_styles):
    style_loss = 0.
    for gen_feat, style_feat, G_style in zip(generated_features.values(), style_features.values(), G_styles.values()):
        _, channel, height, width = gen_feat.shape

        G_gen = gen_feat.view(channel, height * width).mm(gen_feat.view(channel, height * width).t())

        style_loss += torch.mean((G_gen - G_style).pow(2))

    return style_loss
