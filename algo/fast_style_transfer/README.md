We implement a fast style transfer algorithm based on [Perceptual Losses for Real-Time Style Transfer and Super-Resolution]

Although this algorithm is traiend with (256, 256, 3) images, it generalizes extremly well with high-dimensional image at test time. Once you train the network, you can run eval.py to try different scale images.

## Examples

<p align = 'center'>
<img src = 'algo/fast_style_transfer/style/udnie.jpg' height = '246px'>
<img src = 'algo/fast_style_transfer/content/stata.jpg' height = '246px'>
<a href = 'algo/fast_style_transfer/results/stata-udnie.jpg'><img src = 'algo/fast_style_transfer/results/stata-udnie.jpg' width = '627px'></a>
</p>
<p align = 'center'>

<p align = 'center'>
<img src = 'algo/fast_style_transfer/style/udnie.jpg' height = '246px'>
<img src = 'algo/fast_style_transfer/content/chicago.jpg' height = '246px'>
<a href = 'algo/fast_style_transfer/results/chicago-udnie.jpg'><img src = 'algo/fast_style_transfer/results/chicago-udnie.jpg' width = '627px'></a>
</p>
<p align = 'center'>

## Reference Papers

Justin Johnson et al. Perceptual Losses for Real-Time Style Transfer and Super-Resolution

## Reference Code

https://github.com/lengstrom/fast-style-transfer