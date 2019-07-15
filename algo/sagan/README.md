
## Algorithms

- [x] Self-Attention
- [x] Spectral Normalization
- [x] Hinge Loss
- [x] TTUR
- [ ] Conditional Batch Normalization   [code]()
- [ ] Projection Discriminator
- [x] resize for upsampling

## Monitor Training

To monitor your training info, please open tensorboard with `tensorboard --logdir=logs`

## References

Conditional Batch Normalization (CBN): Harm de Vries, Florian Strub, J´er´emie Mary, Hugo Larochelle, Olivier Pietquin, and Aaron C Courville. Modulating early visual processing by language. In NIPS, pp. 6576–6586, 2017.

two-timescale update rule (TTUR): Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In NIPS, pp. 6629–6640, 2017.

Spectral Normalization(SN): Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. Spectral normalization for generative adversarial networks. In ICLR, 2018.

Spectral Norm Regularization: Yuichi Yoshida et al. Spectral Norm Regularization for Improving the Generalizability of Deep Learning

DCGANs: Alec Radford, Luke Metz, Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Network

Conditional Batch Normalization: Harm de Vries, Florian Strub et al. Modulating early visual processing by language

Resize vs Deconvolution: https://distill.pub/2016/deconv-checkerboard/

## Code Reference