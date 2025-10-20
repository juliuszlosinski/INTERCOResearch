ZFNet (2013) is an improved version of AlexNet, developed by Zeiler and Fergus. It refreshed AlexNet's architecture by adjusting hyperparameters. It led to better performance in image classification. The most important improvements were: **smaller first-layer filters (from 11x11 to 7x7)**, **reduced stride size** for linear details, **improved visualization techniques** in ordered to understand how CNNs process images.

**The main architectural and methodological improments included:**
- **Rmaller convolution filters (kernels):** The first-layer size was reduced from 11x11 (used in AlexNet) to 7x7, which allowed the network to capture finer spatial details and preserver more local information early in the processing,
- **Reduced stride size:** The stride in the first convolutional layer was decreased (from 4 to 2) improving the network's ability to model subtle linear and texture details in images,
- **Enhanced visualization techniques:** Zeiler and Fergus introduced **deconvolutional networks (deconvnets)** in order to visualize the activations of intermediate layers. This innovation made it possible to see which image features each layer was responding to, providing deep insights how CNNs process and learn hierarchical features,
- **Improved performance:** Through these optimizations, ZFNet achieved **better classification accuracy** on benchmarks like **ImageNet** compared to previous CNN architecture which was AlexNet (2012).

| **Layer** | **Type**                 | **Filter Size / Stride / Padding** | **# Filters / Units** | **Activation** | **Notes**                                                                             |
| --------- | ------------------------ | ---------------------------------- | --------------------- | -------------- | ------------------------------------------------------------------------------------- |
| 1         | **Convolution**          | 7×7 / 2 / 3                        | 96                    | ReLU           | Smaller filters and reduced stride (vs. AlexNet’s 11×11 / 4) to capture finer details |
| 2         | Max Pooling              | 3×3 / 2                            | —                     | —              | Reduces spatial dimensions                                                            |
| 3         | **Convolution**          | 5×5 / 2 / 2                        | 256                   | ReLU           | Maintains receptive field while improving feature extraction                          |
| 4         | Max Pooling              | 3×3 / 2                            | —                     | —              | Downsampling                                                                          |
| 5         | **Convolution**          | 3×3 / 1 / 1                        | 384                   | ReLU           | Captures more abstract features                                                       |
| 6         | **Convolution**          | 3×3 / 1 / 1                        | 384                   | ReLU           | Further deep feature extraction                                                       |
| 7         | **Convolution**          | 3×3 / 1 / 1                        | 256                   | ReLU           | Final convolutional layer                                                             |
| 8         | Max Pooling              | 3×3 / 2                            | —                     | —              | Spatial dimension reduction before FC layers                                          |
| 9         | Fully Connected          | —                                  | 4096                  | ReLU           | High-level feature representation                                                     |
| 10        | Fully Connected          | —                                  | 4096                  | ReLU           | Same as AlexNet                                                                       |
| 11        | Fully Connected (Output) | —                                  | 1000                  | Softmax        | ImageNet classification output                                                        |
**Key notes:**
- **Built upon AlexNet (2012):**
	ZFNet refined AlexNet's architecture rather than designing a completely new one. The focus was on improving **feature visualization**, **interpretability** and hyperparameter tuning.
- **Smaller filters and stride:**
	- **Filter size:** Reduced from **11x11** to **7x7** in the first convolutional layer.
	- **Stride:** Lowered from **4** to **2**: Allowed the model to capture finer local patterns and edge details, leading to better feature extraction in early layers.
- **Visualization via deconvolutional networks:**
	- Introduced **deconvnets** to **reverse-engineer** the CNN's activations.
	- Helped visualiza which image features each layer was detecting (edges, textures, object parts, etc.),
- **Layer adjustments and hyperparameter tuning:**
	- Slight modifications in layer structure and filter sizes improved representational power,
	- Careful tuning of parameters (strid, padding, filter size) led to **better spatial resolution** in feature maps,
- **Improved ImageNet performance**:
	- Achieved **top-5 error rate of 11,2%** surpassing AlexNet's **16,4%** on the ImageNet 2013 challenge,
	- Schowed that **architecture tunning** and **understanding model behaviour** can yield large gains,
- **Deeper insight into CNN learning:**
	- Demonstrated that deeper layers learn **increasingly abstract representations** (e.g., from edges -> textures -> obejct parts -> full objects),
	- Provided empirical evidence for the **hierarchical feature learning** propery of CNNs.