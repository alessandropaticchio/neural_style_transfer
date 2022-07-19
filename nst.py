from model import VGG_NST
from losses import compute_style_loss, compute_content_loss
from utils import *
from PIL import Image
from torch.optim import Adam
from torchvision.utils import save_image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nst(content_path, style_path, alpha=1, beta=0.01, lr=1e-3, n_steps=1000, image_size=50):
    content_image = Image.open(content_path)
    style_image = Image.open(style_path)

    content_image_name = content_path.split("/")[-1].split(".")[0]
    style_image_name = style_path.split("/")[-1].split(".")[0]

    img_preprocess = get_preprocessing_transform(image_size)
    content_image = img_preprocess(content_image).to(device).unsqueeze(0)
    style_image = img_preprocess(style_image).to(device).unsqueeze(0)

    generated_image = content_image.clone().requires_grad_(True)

    optimizer = Adam([generated_image], lr=lr)

    model = VGG_NST()
    model.eval()

    # Pre-computing style_features and Gram Matrix for style_image
    style_features = model(style_image)
    G_styles = compute_G_style(style_features)

    # Pre-computing content_features
    content_features = model(content_image)

    for step in range(n_steps):
        print("Processing step no. {}".format(step))
        generated_features = model(generated_image)

        content_loss = compute_content_loss(content_features, generated_features)
        style_loss = compute_style_loss(style_features, generated_features, G_styles)

        total_loss = alpha * content_loss + beta * style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()

        if step % 50 == 0:
            print("Total loss so far: {:.4f}".format(total_loss.item()))
            save_image(generated_image, f"data/generated/generated_{content_image_name}_{style_image_name}.png")


if __name__ == '__main__':
    content_path = "data/content/io.jpg"
    style_path = "data/style/swirls.jpeg"
    n_steps = 6000
    alpha = 1
    beta = 1
    lr = 1e-3
    image_size = 500

    nst(content_path=content_path, style_path=style_path, n_steps=n_steps, alpha=alpha, beta=beta, lr=lr,
        image_size=image_size)
