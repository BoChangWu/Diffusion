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
