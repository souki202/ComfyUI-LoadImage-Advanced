# ComfyUI-LoadImage-Advanced

主にLoadImageとVae Encode、Upscaleを統合したノードです。

![sample](./images/sample.png)

## 機能

- 1つのノードにまとめられたload image, upscale, vae encode
- 解像度を自動的に任意のnの倍数に補正する
  - 読み込んだ画像の解像度がnの倍数でないことによるエラーを自動で防止できます
  - fluxを使用する場合は16、他は16か8あたりが推奨です

## Functions

- Load image, upscale, vae encode in one node
- Automatically adjust the resolution to any multiple of n
  - Automatically prevents errors due to the resolution of the loaded image not being a multiple of n
  - 16 is recommended when using flux, 16 or 8 for others
