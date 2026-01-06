GoogLeNet (Inception v1, 2014) is a convolutional neural network (CNN) architecture developed by researchers at Google. It introduced the **Inception module**, which made the network more computationally efficient while maintaining high accuracy.

**The key features include:**
- The inception module applies **multiple convolution filters of different sizes (1x1, 3x3, 5x5)** in parallel to capture features at multiple scales.
- **1x1 convolutions** are used for **dimensionality reduction**, decreasing the number of parameters and computation before applying larger filters,
- It is **deep network (22 layers)** but highly optimized for efficiency, containing fat fewer parameters (~5 million) compared to VGGNet (~138 million),
- It replaces normal fully connected layers with global average pooling, reducing overfitting and further lowering parameter count.

**The Inception family** evolved through several versions:
- **Inception v1 (2014)** - introduced the original Inception module with multi-scale filters,
- **Inception v2 (2015)** - improved training with **batch normalization** and **factorized convolutions**,
- **Inception v3 (2015)** - further optimized efficiency by **replacing 5x5 convolutions with two 3x3 convolutions**,
- **Inception v4 (2016)** - combined **Inception modules with residual connections**, resulting in **Inception-ResNet**, which achieved ever better accuracy.

**Key notes:**
- **Total depth:** 22 layers (not counting pooling or softmax),
- **Inception module:** Combines parallel branches of 1x1, 3x3 and 5x5 convolutions + pooling -> concatenated depth-wise,
- **1x1 convolutions** are used to **reduce dimensionality** before 3x3 and 5x5 filters -> drastically lowers computation,
- **Global average pooling**: Replaces fully connected layers to reduce overfitting and parameter count (~5 million parameters),
- **Auxiliary classifiers:** Two intermediate classifiers (at Inception 4a and 4d) help gradient flow during training (used only during training).

**History:**
- 1. Inception v1 (GoogLeNet, 2014):
	- **Paper:** "Going Deeper with convolutions" (CVPR 2015),
	- **Key idea:** Introduced the Inception module, allowing multiple filter sizes (1x1, 3x3, 5x5) in parallel to capture multi-scale information efficiently.
	- **Key innovation:** 1x1 convolutions for dimensionality reduction.
	- **Dataset:** ImageNet 2014 winner.
- 2. Inception v1 (2015):
	- **Paper:** "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
	- **Main feature:**
		- Added Batch Normalization (BN) to Inception v1, improving training stability and convergence.
		- Sometimes referred to as "Inception v1 + BN".
- 3. Inception v3 (2015-2016):
	- **Paper:** "Rethinking the Inception Architecture for Computer Vision"
	- **Main feature:**
		- Introduced more efficient factorization of convolutions (such as 1x1 convolutions), reducing computational cost withou sacrificing accuracy,
		- Used label smoothing for regularization, which helps to prevent overfitting,
		- Improved architecture design for both computational efficiency and performance,
		- Included the use of auxiliary classifiers to provide additional gradient signals during training.