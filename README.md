# nn-hallucinations

Repeated attempts at teaching NNs how to be more creative than me. Contains tools and algorithmic hallucinigens. 

## Implemented Algorithms and Observations

*Note: for results TODO

### MADE

Code: https://github.com/EugenHotaj/nn-hallucinations/blob/master/MADE.ipynb <br>
Paper: http://proceedings.mlr.press/v37/germain15.pdf <br>
Best NLL: **85.** (mine) 88.7 (paper - 1h1

#### Input ordering
The main MADE algorithm is agnostic to the ordering of the dimensions fo the input and even permutes the ordering each time it samples a new mask. My initial intuition was that using natural ordering (i.e. left-to-right, top-to-bottom) for images would perform equivalently to (if not better) a random permutation. After all, images have definite structure to them which is part of the reason why Convolutional Networks work so well. However, randomly permuting the ordering noticably outperforms the natural ordering. 

My hypothesis is that, at least for handwritten digits, it's much easier to guess the value of a pixel at some position given the values of pixels at other locations than it is to guess the value given the previous pixels in the natural ordering. For example, consider the digit `7`. Suppose you're trying to guess the value of the pixel on the middle-left of the image directly under the hanging bit of the `7`. In the natural ordering, all you would have seen is the top bar of the digit `7` and looking at just the top bar it might be hard to correctly guess what the value of the pixel under consideration is. For example, the image could be an `8` or a `9`, in which case the pixel might be black. Or it could be a `3` or a `7`, in which case the pixel might be white. 

On the other hand, if you have access to a smattering of random pixels, TODO...

#### Sample quality
TODO

### NADE

Code: https://github.com/EugenHotaj/nn-hallucinations/blob/master/NADE.ipynb <br>
Paper: http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf 

### Neural Style Transfer

Code: https://github.com/EugenHotaj/nn-hallucinations/blob/master/NADE.ipynb <br>
Paper: https://arxiv.org/pdf/1508.06576.pdf

###

Code: https://github.com/EugenHotaj/nn-hallucinations/blob/master/NADE.ipynb <br>
Background: https://en.wikipedia.org/wiki/Compositional_pattern-producing_network
