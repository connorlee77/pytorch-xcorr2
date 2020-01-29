# pytorch-xcorr2
Batchwise zero normalized cross correlation for pytorch.

This extends the correlation layer in https://github.com/rafellerc/Pytorch-SiamFC. Correlations between images of the same size are much faster by using a dot product instead of a convolution. 

#### Usage:
```python
correlate = xcorr2(zero_mean_normalize=True)

img1 = torch.rand(BATCH_SIZE, C, H, W)
img2 = torch.rand(BATCH_SIZE, C, H, W)

scores = correlate(img1, img2)
```
