VGGNet (2014) is CNN architecture developed by the Visual Geometry Group at Oxford University. It improved upon previous models by using **deeper architecture** with small **3x3** convolutional filters (kernels). The most key features are: **increased depth (16 or 19 layers)** for better feature extraction, **uniform 3x3 filters (kernels)** instead of larger ones for improving learning capacity, **max pooling layers** to reduce spatial dimensions while preserving key features, **fully connected layers** at the end for classification. It has high computational cost. VGGNet comes in two main variants:
- **VGG-16** contains 16 weight layers:
	- 13 convolutional (CNN),
	- 3 fully connected (FC).
- **VGG-19** contains 19 weight layers:
	- 16 convolutional (CNN),
	- 3 fully connected (FC).
Both models follow the same design principle, using only **3x3 convolutional filters/kernels** and **max pooling layers**. VGG-19 have more convolutional layers.

**Overview:**
**VGGNet (Visual Geometry Group Network)** was introduced in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" by Karen Simonyan and Andrew Zisserman from the University of Oxford. It was submitted to the **ILSVRC 2014 (ImageNet Large Scale Visual Recognition Challange)** and became one of the most influential CNN architectures in deep learning.

**Key architecture design:**
- **Depth as the key innovation:** Prior CNNs like **AlexNet (2012)** and **ZFNet (2013)** had 8 layers, while **VGGNet** showed that increasing depth (up to 16 or 19 layers) significantly improves performance.
- **Small, uniform convolution filters:** Every convolutional layer uses **3x3 filters** (the smallest size that can capture spatial relationships) with stride of 1 and padding of 1, preserving spatial resolution.
- **Stacked convolutions for effective receptive field:** Stacking two of three 3x3 convolutions has the same receptive field as 5x5 or 7x7 filter but with fewer parameters and more non-linearities improving the model's representational power.
- **Max pooling:** After every few convolutional layers, **2x2 max pooling** **with stride 2** is applied to reduce the spatial dimensions while keeping essential features.
- **Fully connected layers:** At the end, VGGNet includes **three fully connected (FC) layers** - the first two have 4096 neurons each, an the last one outputs class scores (e.g. **1000 for ImageNet**).
- **Softmax classifier:** The final layer uses a softmax function for classfication probabilites.

**Variants:**
- **VGG-16:** 13 convolutional + 3 fully connected layers,
- **VGG-19:** 16 convolutional + 3 fully connected layers.

Both share the same general layout:
`[Conv → Conv → Pool] × n → FC → FC → FC → Softmax`

**Model characteristics:**
- **Parameters:** Very large around **138 milion parameters** for VGG-16. This makes it computationally expensive and memory-intensive, requiring powerful GPUs for training.
- **Input size:** The network expects 224x224 RGB images,
- **Pre-training:** Commonly used as as feature extractor in transfer learning due to its simplle and uniform architecture.

**Performance:**
- Achieved **top-5 accuracy of 7.3%** on ImageNet (2014),
- Demonstrated that **depth** was a crucial factor for CNN performance, influencing future architectures like **ResNet, Inception and DenseNet**.

**Pros and cons:**
- **Advantages:**
	- Simple and elegant architecture,
	- Demonstrated that depth improves accuracy,
	- Excellent for feature extraction and transfer learning,
- **Disadvantages:**
	- High memory and computational requirements,
	- Large number of parameters,
	- Slower inference compared to newer and more efficient networks.