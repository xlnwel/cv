We implement a fast style transfer algorithm based on [Perceptual Losses for Real-Time Style Transfer and Super-Resolution]

Although this algorithm is traiend with (256, 256, 3) images, it generalizes extremly well with high-dimensional image at test time. Once you train the network, you can run eval.py to try different scale images.

## Examples

<p align = 'center'>
<img src = 'data/style/udnie.jpg' height = '246px'>
<img src = 'data/content/stata.jpg' height = '246px'>
<a href = 'data/results/stata-udnie.jpg'><img src = 'data/results/stata-udnie.jpg' width = '627px'></a>
</p>
<p align = 'center'>

<p align = 'center'>
<img src = 'data/style/udnie.jpg' height = '246px'>
<img src = 'data/content/chicago.jpg' height = '246px'>
<a href = 'data/results/chicago-udnie.jpg'><img src = 'data/results/chicago-udnie.jpg' width = '627px'></a>
</p>
<p align = 'center'>

## Reference Papers

Justin Johnson et al. Perceptual Losses for Real-Time Style Transfer and Super-Resolution

## Reference Code

https://github.com/lengstrom/fast-style-transfer