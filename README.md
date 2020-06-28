# nn-hallucinations

Repeated attempts at teaching NNs how to be more creative than me.

## Implemented Algorithms and Observations

*Note: MADE and NADE results are more optimistic in our implementation. Instead of using a fixed dataset, we resample a new binary MNIST dataset on every epoch. We can think of this as using data augmentation which helps our models learn better.*

### MADE

Code: https://github.com/EugenHotaj/nn-hallucinations/blob/master/MADE.ipynb <br>
Paper: http://proceedings.mlr.press/v37/germain15.pdf <br>
Best NLL: **84.867** (ours w/ 1hl, 1 mask) 88.04 (paper w/ 1hl, 32 masks)

#### Input ordering
The MADE algorithm is agnostic to the ordering of the dimensions of the input and randomly permutes the ordering on each batch. Our initial intuition was that using natural ordering (e.g. left-to-right, top-to-bottom for images) would perform equivalently to random permutations. However, natural ordering seems to greatly underperform random permutations. 

Our hypothesis is that random pixels convey more information about an image than pixels in the natural ordering, at least for binary MNIST digits. As a simple thought experiment, consider the digit `7`. Suppose we're trying to predict a pixel directly under the hanging bar of the `7`. Given the natural ordering, we only know that the image contains a long bar at the top, while the rest of the image is ambiguous. For example, the top bar could be part of the loop in the digits `8` or `9`, or it could be the top bar of the digits `5` or `6`. On the other hand, by observing random pixels, we may see that there are no black pixels on the lower left side of the image (ruling out `8` and `6`), no black pixels on the lower right of the image (ruling out `5` and `9`), and so on. 

#### Masks
Using more than 1 mask seems to heavily regularize the MADE model with no apparent benefits (at least when training for the same number of epochs). Even in the paper, the authors only gain .36 NLL by using 32 masks and introducing an "ensembling" procedure at test time. The complexity of this approach seemed to outweigh the benefits so we did not implement it.

#### Sample quality
Even though MADE is able to achieve lower NLL than NADE (albeit with *16x* more neurons in the hidden layer), NADE seems to generate (subjectively) more pleasing samples. This matches what is reported in [Theis et al. (2016)](https://arxiv.org/abs/1511.01844), namely that good NLL does not necessarily lead to higher quality samples. 

### NADE

Code: https://github.com/EugenHotaj/nn-hallucinations/blob/master/NADE.ipynb <br>
Paper: http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf <br>
NLL: **85.65** (ours) 88.86 (paper)

We used the Adam optimizer instead of plain SGD, which may account for part of the improvement we see over the paper results. 

### Neural Style Transfer

Blog: https://towardsdatascience.com/how-to-get-beautiful-results-with-neural-style-transfer-75d0c05d6489 <br>
Code: https://github.com/EugenHotaj/nn-hallucinations/blob/master/NADE.ipynb <br>
Paper: https://arxiv.org/pdf/1508.06576.pdf

### Compositional Pattern Producing Networks

Code: https://github.com/EugenHotaj/nn-hallucinations/blob/master/NADE.ipynb <br>
Background: https://en.wikipedia.org/wiki/Compositional_pattern-producing_network
