# 8. Convolutional Neural Networks

## 8.1 From FC to Convolutions

- **Translation invariance:** In the earliest layers, our network should respond similarly to the same patch, regardless of where it appears in the image.

  This implies that a shift in the input X should simply lead to a shift in the hidden representation H. This is only possible if V and U do not actually depend on (i,j)


- **Locality principle**: The earliest layers of the network should focus on local regions, without regard for the contents of the image in distant regions

  We should not have to look very far away from location (i,j) in order to glean relevant information to assess what is going on at [H]i,j.

![](imgs/conv.png)

Convolutional neural networks (CNNs) are a special family of neural networks that contain convolutional layers. In the deep learning research community, V is referred to as a convolution kernel, a filter, or simply the layer’s weights that are often learnable parameters.

## 8.2 Convolutions for Images

**The Cross-Correlation Operation**:

![](imgs/cc.png)

Note that along each axis, the output size is slightly smaller than the input size.

Output size = (n_h - k_h + 1) x (n_w - k_w + 1)

The convolutional layer output is sometimes called a feature map, as it can be regarded as the learned representations (features) in the spatial dimensions (e.g., width and height) to the subsequent layer.

**Conv layer:** A convolutional layer cross-correlates the input and kernel and adds a scalar bias to produce an output. The two parameters of a convolutional layer are the kernel and the scalar bias. When training models based on convolutional layers, we typically initialize the kernels randomly, just as we would with a fully-connected layer.

## 8.3 Padding and stride

**Padding:** one tricky issue when applying convolutional layers is that we tend to lose pixels on the perimeter of our image. One straightforward solution to this problem is to add extra pixels of filler around the boundary of our input image.

![](imgs/padding.png)

Output size = (n_h - k_h + p_h + 1) x (n_w - k_w + p_w + 1)

In many cases, we will want to set: *p_h = k_h - 1* to give the input and output the same height and width.

**Stride:** we refer to the number of rows and columns traversed per slide as the stride. The stride can reduce the resolution of the output, for example reducing the height and width of the output to only 1/n of the height and width of the input (n is an integer greater than 1).

So, putting it all together, our output will be:

![](imgs/output.png)

## 8.4 Multiple input and Multiple Output channels

**Multiple inputs:** when channels > 1, we need a kernel that contains a tensor of shape kh × kw for every input channel. Concatenating these ci tensors together yields a convolution kernel of shape ci × kh × kw. Since the input and convolution kernel each have ci channels, we can perform a cross-correlation operation on the two-dimensional tensor of the input and the two-dimensional tensor of the convolution kernel for each channel, adding the ci results together (summing over the channels) to yield a two-dimensional tensor. This is the result of a two-dimensional cross-correlation between a multi-channel input and a multi-input-channel convolution kernel.

![](imgs/mi.png)

**Multiple outputs:** Denote by ci and co the number of input and output channels, respectively, and let kh and kw be the height and width of the kernel. To get an output with multiple channels, we can create a kernel tensor of shape ci×kh×kw for every output channel. We concatenate them on the output channel dimension, so that the shape of the convolution kernel is co×ci×kh×kw.

**1x1 Conv layer:** You could think of it as constituting a fully-connected layer applied at every single pixel location to transform the ci corresponding input values into co output values.

![](imgs/one.png)

Is typically used to adjust the number of channels between network layers and to control model complexity.

## 8.5 Pooling

## 8.6 LeNet
