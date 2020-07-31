# pytorch-generative
*Repeated attempts at teaching NNs how to be more creative than myself.*

## Autoregressive Generative Models

Binary MNIST (NLL): 

| Algorithm | Our Results | Best Other Results | Links |
| --- | ---| --- | --- |
| Gated PixelCNN | 81.50 | **81.30** \[1\] | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/gated_pixel_cnn.py), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/gated_pixel_cnn.ipynb) |
| PixelCNN | 81.45 | **81.30** \[1\] | [Code](), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/pixel_cnn.ipynb) |
| MADE | **84.867** | 88.04 \[4\]| [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/made.py), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/MADE.ipynb) |
| NADE | **85.65** | 88.86 \[5\] | [Code](https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/nade.py), [Notebook](https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/NADE.ipynb) |

*Note:* Our reported binary MNIST results may be optimistic. Instead of using a fixed dataset, we resample a new binary MNIST dataset on every epoch. We can think of this as using data augmentation which helps our models learn better.

### Notes

Below we list notes... TODO

#### Gated PixelCNN

We were not able to find binary MNIST results for the Gated PixelCNN model so we compare to the original PixelCNN model in \[1\]. In theory, Gated PixelCNN should be a stronger model than the original PixelCNN, although we were not able to beat PixelCNN (though we did get close). We have a few hypotheses for why this may be the case:

TODO.

#### MADE

**Input ordering** </br>
The MADE algorithm is agnostic to the ordering of the dimensions of the input and randomly permutes the ordering on each batch. Our initial intuition was that using natural ordering (e.g. left-to-right, top-to-bottom for images) would perform equivalently to random permutations. However, natural ordering seems to greatly underperform random permutations. 

Our hypothesis is that random pixels convey more information about an image than pixels in the natural ordering, at least for binary MNIST digits. As a simple thought experiment, consider the digit `7`. Suppose we're trying to predict a pixel directly under the hanging bar of the `7`. Given the natural ordering, we only know that the image contains a long bar at the top, while the rest of the image is ambiguous. For example, the top bar could be part of the loop in the digits `8` or `9`, or it could be the top bar of the digits `5` or `6`. On the other hand, by observing random pixels, we may see that there are no black pixels on the lower left side of the image (ruling out `8` and `6`), no black pixels on the lower right of the image (ruling out `5` and `9`), and so on. 

**Masks** </br>
Using more than 1 mask seems to heavily regularize the MADE model with no apparent benefits (at least when training for the same number of epochs). Even in the paper, the authors only gain .36 NLL by using 32 masks and introducing an "ensembling" procedure at test time. The complexity of this approach seemed to outweigh the benefits so we did not implement it.

**Sample quality** </br>
Even though MADE is able to achieve lower NLL than NADE (albeit with *16x* more neurons in the hidden layer), NADE seems to generate (subjectively) more pleasing samples. This matches what is reported in [Theis et al. (2016)](https://arxiv.org/abs/1511.01844), namely that good NLL does not necessarily lead to higher quality samples. 

#### NADE
We used the Adam optimizer instead of plain SGD, which may account for part of the improvement we see over the paper results. 

### References

1. https://arxiv.org/pdf/1601.06759.pdf 
1. https://arxiv.org/abs/1606.05328
1. http://www.scottreed.info/files/iclr2017.pdf
1. https://arxiv.org/abs/1502.03509 
1. http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf



## Neural Style Transfer
Blog: https://towardsdatascience.com/how-to-get-beautiful-results-with-neural-style-transfer-75d0c05d6489 <br>
Notebook: https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/style_transfer.ipynb <br>
Paper: https://arxiv.org/pdf/1508.06576.pdf

## Compositional Pattern Producing Networks
Notebook: https://github.com/EugenHotaj/pytorch-generative/blob/master/notebooks/cppn.ipynb <br>
Background: https://en.wikipedia.org/wiki/Compositional_pattern-producing_network
