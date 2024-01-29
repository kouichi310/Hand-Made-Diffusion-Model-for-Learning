from datasets import load_dataset
from matplotlib import pyplot as plt # hugging faceのdatasetを使う

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.optim import Adam

from pathlib import Path
import torch

import Unet
import util


def transforms_data(examples):
    #画像データを数値データにする
    examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
    del examples["image"]
    return examples

dataset = load_dataset("fashion_mnist")

image_size = 28
channels = 1
batch_size = 128

transform = Compose([
    transforms.RandomHorizontalFlip(), #ランダムに左右を反転させデータを増やす
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

transformed_dataset = dataset.with_transform(transforms_data).remove_columns("label")

dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

results_folder = Path("result/generation")
results_folder.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet.Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
    resnet_block_groups=4,
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

epochs = 10
timesteps = 200
betas = util.linear_beta_schedule(timesteps=timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) #分散
sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # 標準偏差

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device) # データを設定

        t = torch.randint(0, timesteps, (batch_size,), device=device).long() # タイムステップ情報をバッチごとにランダムに与える
        loss = util.p_losses(model, batch, t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod) # 画像を生成し損失を計算

        if step % 100 == 0: # 表示
            print("Loss", loss.item())

        loss.backward() # 勾配の計算
        optimizer.step() # パラメータの更新

    # 画像の生成
    samples = util.sample(model, timesteps=timesteps,image_size=image_size, batch_size=64, channels=channels,beta=betas, posterior_variance=posterior_variance, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas=sqrt_recip_alphas)
    save_image(torch.from_numpy(samples[-1]), str(results_folder / f'sample-{epoch}.png'), nrow=5)

samples = util.sample(model, timesteps=timesteps, image_size=image_size, batch_size=64, channels=channels,beta=betas)
random_index = 1
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels).squeeze(), cmap="gray")
