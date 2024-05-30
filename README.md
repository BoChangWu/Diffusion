# Image Generating

圖像生成是生成式AI廣為人知的領域，網路上有許多成功的案例如Dall·E2、Imagen、MidJourney，他們都是使用Diffusion Model做的來進行圖像生成。甚至在OpenAI發表的Sora都使用了Diffusion與Transformer結合，相信Diffusion一定有神奇的地方才能這麼廣泛使用，因此我基於好奇想要研究其中的原理。

## 什麼是DDPM
DDPM(Denoising Diffusion Probabilistic Models)主要分為forward process以及 reverse process兩個部分，下面將分別說明這兩個部分。

### Forward Process
forward process的概念是將一張清楚的照片不斷添加noise(雜訊)，一直到整張圖片都是noise且noise呈現高斯分布。

假設對圖片添加了N次noise，且每一次noise都標記第t次noise，那麼我們就會得到noise過程中的所有紀錄(noise、步驟、結果)。


![alt text](/img/image-2.png)

### Reverse Process
reverse process則是將一張呈現高斯分佈的noise圖不斷地denoise(去雜訊)，一直noise變成一張乾淨清楚的圖片為止。

![alt text](/img/image-3.png)

## Implentation
有了Diffusion的概念，對於實際上是如何實行的還是有點模糊，因此接下來要說明模型的結構，在這個diffusion模型中，主要會使用到兩個部分，一是 forward process， 有了forward process，才能得到訓練時所需要的noise和step；再來是U-Net，由Unet作為reverse process時的預測模型有相當好的效果。

Forward process 在前面有提及，簡單來說就是將圖片不斷地加上noise

```
def forward_diffusion(x_0, t, betas = torch.linspace(0.0, 1.0, times)):
    noise = torch.randn_like(x_0)
    alphas = 1 - betas
    alphas_hat = torch.cumprod(alphas, axis=0)
    alphas_hat_t = alphas_hat.gather(-1, t).reshape(-1, 1, 1, 1)
    
    mean = alphas_hat_t.sqrt() * x_0
    variance = torch.sqrt(1 - alphas_hat_t) * noise
    
    return mean + variance, noise
```

#### 為什麼要使用U-Net
U-Net是一種特殊類型的卷積神經網路（CNN），最初是設計來用於segmentation task(圖像分割任務)，但由於U-Net特殊的結構設計，也被廣泛用於其他生成任務當中。在Diffusion中，U-Net的主要功能是denoise並重構圖片。

![alt text](/img/unet.png)

在訓練過程中，U-Net會需要在forward process過程中第$t$步及其結果$X_t$作為輸入，透過運算得到第$t$步的noise作為output，接著將X_t去除noise就可以得到第t-1步的結果$X_(t-1)$，這正是我們reverse process的過程。

## 訓練結果