import numpy as np
from tqdm.auto import tqdm

# PyTorch, 計算関係
import torch
import torch.nn.functional as F

# 描画用
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize


def extract(a, t, x_shape):
    batch_size = t.shape[0] # バッチサイズ
    out = a.gather(-1, t.cpu()) # aの最後の次元 ⇒ timestepに対応するalphaを取ってくる
    return out.reshape(batch_size, *((1,) * (len(x_shape) -  1))).to(t.device) # バッチサイズ x 1 x 1 x 1にreshape

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    # 綺麗な画像からノイズを加えた画像をサンプリングする
    if noise is None:
        noise = torch.randn_like(x_start)

    #t時点での平均計算
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    #t時点での分散計算用
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, reverse_transform):
    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod) # キレイな画像にタイムステップを渡す
    #0次元目に追加したバッチ用の次元を削除
    noisy_image = reverse_transform(x_noisy.squeeze())

    return noisy_image

def p_losses(denoise_model, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="huber"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                        noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.l2_loss(noise, predicted_noise)
    elif loss_type == 'huber':
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index, betas, posterior_variance, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas):
    #beta_t
    betas_t = extract(betas, t, x.shape)
    # 1 - √\bar{α}_t
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)

    # 1 / √α_t
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # μ_Θをモデルで求める: model(x, t)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        # σ^2_tを計算
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        # 正規乱数zをサンプリング
        noise = torch.rand_like(x)

    # x_{t-1}
    return model_mean + torch.sqrt(posterior_variance_t) * noise

#no_gradは勾配計算用のパラメータである自動微分を保存しない設定（メモリ消費削減）
@torch.no_grad()
def p_sample_loop(model, shape, timesteps, beta, posterior_variance, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas):
    device = next(model.parameters()).device

    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i,betas=beta, posterior_variance=posterior_variance, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas=sqrt_recip_alphas)
        imgs.append(img.cpu().numpy())

    return imgs

@torch.no_grad()
def sample(model, image_size, timesteps,beta, posterior_variance, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, batch_size=16, channels=3):
    return p_sample_loop(model,beta=beta, shape=(batch_size, channels, image_size, image_size),timesteps=timesteps, posterior_variance=posterior_variance, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas=sqrt_recip_alphas)


if __name__ == "__main__":
    timesteps = 200
    betas = linear_beta_schedule(timesteps=timesteps)

    alphas = 1. - betas

    # 入力テンソルの累積積(テンソルは多次元データを表す。ベクトルや行列もテンソルの一部。)
    # 画像のテンソルは、幅、高さ、色情報の3つのテンソルで表現できる
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    #平均・分散を計算するため√(α¯_{t}),と1−α¯tを計算しておきます。
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    data = 'data/chiba.jpg'
    image = Image.open(data)

    # pytorchで処理するため、画像データをテンソルに変換
    image_size = 128
    transform = Compose([
        Resize(image_size), #サイズの縮小
        CenterCrop(image_size), #128*128にする
        ToTensor(), #テンソルにする[0,1]
        Lambda(lambda t: (t *2) - 1) #[-1,1]にする
    ])

    # 変換後をx_start
    x_start = transform(image).unsqueeze(0) # 0次元にバッチ用の次元を追加

    #ノイズを加えた画像を描画する前処理
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2), # [-1, 1]を[0, 1]に変換
        Lambda(lambda t: t.permute(1, 2, 0)), # CHW => HWC (チャネル、高さ、幅を高さ、幅、チャネルに変換)
        Lambda(lambda t: t * 255.), # [0, 1] ⇒ [0, 255]
        Lambda(lambda t: t.numpy().astype(np.uint8)), # 整数に変換
        ToPILImage(), # PILの画像に変換
    ])

    # forward process
    t = torch.tensor([199])
    noisy_image = get_noisy_image(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, reverse_transform)
    save_path = f"result/noisy_image_t_199.png"
    noisy_image.save(save_path)

    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) #分散
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # 標準偏差


