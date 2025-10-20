**ResNet (Residual Network)** from 2015 is a convolutional neural network architecture created by Microsoft Corporation. It introduced **residual connections (skip connections)** to solve the problem of **vanishing gradients**, allowing much deeper networks to be trained effectively. The most important features are: **residual connections** skip layers in order to allow direct gradient flow preventing degradation in deep networks, **very deep architecture** can have 50/101/152 oraz even 1000+ layers, **bootleneck** layers (1x1 convolutions) reduce computation while maintaining accuracy, **batch normalization** in order to have stable and faster training. 
Common ResNet variants are:
- **ResNet-18** and **ResNet-34** shallower models for smaller tasks,
- **ResNet-50** uses bottleneck layers for better efficiency,
- **ResNet-101** and **ResNet-152** for deeper networks for high-accuracy tasks,
- **ResNeXt** an improved version with **GROUPED CONVOLUTIONS**.

### **1. Original ResNet (2015)**

**Paper:** "Deep Residual Learning for Image Recognition"  
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
#### **Key Features:**
- **Residual Learning**: The main idea was to allow shortcuts or skip connections in the network, which helped mitigate the vanishing gradient problem. This allowed very deep networks (up to 152 layers) to be trained effectively.
- **Residual Block**: The basic building block of ResNet is the **residual block**, where the input to a layer is added to its output, enabling the network to learn residual mappings rather than the direct mapping.
- **Performance**: ResNet-50, ResNet-101, and ResNet-152 were the key variants, with ResNet-50 being the most widely used due to its balance of depth and computational efficiency.
- **State-of-the-art Results**: ResNet-152 achieved **first place in the ILSVRC 2015 (ImageNet competition)**, beating the previous champion (Inception v3) by a significant margin.
#### **Popular Variants:**
- **ResNet-18**: 18 layers, lighter model for faster training and simpler tasks.
- **ResNet-34**: 34 layers, another relatively shallow version.
- **ResNet-50**: 50 layers, popular due to its balance of depth and performance.
- **ResNet-101**: 101 layers, deeper for more complex tasks.
- **ResNet-152**: 152 layers, the deepest version in the original paper.

### **2. ResNet v2 (2016)**

**Paper:** "Identity Mappings in Deep Residual Networks"  
**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
#### **Key Features:**
- **Improvement to Residual Connections**: ResNet v2 introduced **pre-activation** residual blocks. In the original ResNet, the activation was applied after the convolutional layers; in v2, the activation was applied before the convolution. This small change improved training and performance.
- **Better Gradient Flow**: The change in architecture improved the gradient flow during backpropagation, making the network easier to train.
- **Performance Gains**: ResNet v2 models showed improved accuracy over the original ResNet models, even with fewer layers.
#### **Popular Variants of ResNet v2:**
- **ResNet-50 v2**: 50 layers with pre-activation residual blocks.
- **ResNet-101 v2**: 101 layers with the improved structure.
- **ResNet-152 v2**: 152 layers with the enhanced pre-activation residuals.

### **3. ResNeXt (2017)**

### **ResNeXt: Aggregated Residual Transformations for Deep Neural Networks (2017)**

**Paper:** "Aggregated Residual Transformations for Deep Neural Networks"  
**Authors:** X. Xie, R. Girshick, P. Dollar, Z. Tu, K. He  
**Published:** 2017

---

**Introduction to ResNeXt:**

**ResNeXt** is an advanced variant of the **Residual Network (ResNet)**, introduced to improve performance through a simple yet effective change in the network architecture: the introduction of **cardinality**. Cardinality refers to the **number of paths** (or transformations) in a residual block. By increasing the number of paths, ResNeXt can learn more complex features while keeping the number of parameters low. This architecture essentially takes the idea of residual learning (from ResNet) and adds another layer of flexibility, combining multiple transformations in parallel to create richer feature representations.

ResNeXt was designed to achieve **better accuracy** than traditional ResNet models with **fewer parameters** and **computational resources** by leveraging **grouped convolutions** and **cardinality**.

---

### **Key Concepts of ResNeXt:**

1. **Residual Learning with Cardinality:**
    - **Cardinality** is the number of different transformations (or paths) that are aggregated in each residual block. In standard ResNet, each residual block only has one path, where the input is directly added to the output of the convolution layers. In ResNeXt, multiple transformations are performed in parallel and then aggregated (summed up).
    - This concept of cardinality allows the network to explore multiple parallel learning pathways and enables the model to represent more complex relationships in the data.
        
2. **Grouped Convolutions:**
    - ResNeXt uses **grouped convolutions**, a concept popularized by the **AlexNet** and **VGG** networks, which divides the input channels into several groups and performs convolutions within each group separately. By doing so, ResNeXt is able to reduce the computational burden while still achieving improved representational power.
    - The key here is that multiple groups of convolutions can be executed simultaneously, allowing for greater diversity in the learned features without significantly increasing the computational load.
        
3. **ResNeXt Block:**
    - The **ResNeXt block** is the core building block of the network. It consists of several paths (determined by cardinality) that each perform the same transformation (such as a set of convolutions), and these outputs are **aggregated** by a summation operation. The network’s final output is the sum of these aggregated features.
    - This parallel structure allows for the simultaneous learning of multiple, diverse feature sets, enriching the model’s ability to capture complex patterns in the data.
        
4. **Bottleneck Design:**
    - Similar to **ResNet**'s **bottleneck design** (used in deeper networks like ResNet-50 and ResNet-152), ResNeXt employs a **bottleneck** architecture to reduce the number of parameters. The bottleneck consists of three operations:
        - A **1x1 convolution** to reduce the dimensionality of the input.
        - A **3x3 convolution** (or group convolution) to extract features.
        - A **1x1 convolution** to restore the dimensionality.
    - This efficient design ensures that ResNeXt can maintain both depth and complexity while keeping the number of parameters manageable.
        
5. **Performance Efficiency:**
    - The key advantage of ResNeXt over traditional ResNet models is its ability to achieve **high accuracy** with fewer parameters and reduced computational cost. This is due to the use of **cardinality** and **group convolutions**, which allow for richer feature representations without significantly increasing the number of computations required.
    - ResNeXt’s approach to combining multiple convolutions in parallel means it can explore a more extensive set of feature maps without significantly increasing the parameter count, leading to more efficient training and inference.

---
### **ResNeXt Architecture Overview:**

1. **Residual Blocks and Aggregated Transformations:**
    - Each **ResNeXt block** performs several convolutions in parallel (defined by the cardinality). The output of these parallel convolutions is summed up before being passed through the next layer. This aggregation of multiple transformations allows ResNeXt to exploit diversity in feature learning while keeping the architecture simple.
        
2. **Scalability:**
    - The cardinality parameter can be tuned to scale the model up or down. Larger cardinality increases the diversity of transformations, leading to better learning, but it also increases computational cost. Finding the optimal balance between cardinality and computational efficiency is key to achieving high performance with ResNeXt.
        
3. **Depth vs. Cardinality:**
    - ResNeXt shows that **cardinality** is more effective for improving model performance than increasing the depth of the network. Although increasing depth generally improves accuracy, it also increases the risk of overfitting and computational burden. In contrast, increasing cardinality allows for more powerful feature learning without the downsides of deeper networks.
        
4. **Residual Connections:**
    - The **skip connections** in ResNeXt allow gradients to flow more easily during backpropagation, addressing the vanishing gradient problem that often plagues very deep networks. This makes ResNeXt particularly effective for training very deep networks without losing the benefits of residual learning.

---

### **Why ResNeXt Works:**

1. **Efficient Representation Learning:**
    - By aggregating multiple transformations, ResNeXt can capture a more diverse set of features from the input data. Each transformation (or path) in the residual block learns different aspects of the data, which are then aggregated to form a rich and comprehensive feature representation.
        
2. **Increased Capacity with Low Computational Overhead:**
    - ResNeXt increases the capacity of the network through **cardinality**, but this increase is more computationally efficient compared to simply making the network deeper. The grouped convolutions and the parallel nature of the transformations allow for increased representational power without a significant increase in parameter count or computational load.
        
3. **Versatility:**
    - ResNeXt can be applied to a variety of tasks like image classification, object detection, and segmentation. The architecture scales well and can be adapted to different types of computer vision problems.
        
4. **Parameter Efficiency:**
    - Due to the use of **grouped convolutions** and **cardinality**, ResNeXt achieves high accuracy with fewer parameters than traditional deep architectures. This makes it a suitable choice for applications where computational resources are limited, such as on embedded systems or mobile devices.

---

### **Performance of ResNeXt:**

- **Image Classification:** ResNeXt outperforms traditional ResNet models in **ImageNet** classification tasks. By increasing cardinality, ResNeXt achieves a higher accuracy than ResNet models with a similar or even smaller number of parameters.
- **Computational Efficiency:** Despite being deeper and more complex than ResNet, ResNeXt is computationally efficient due to its ability to combine multiple convolutions in parallel. This makes it suitable for large-scale tasks where computational efficiency is a concern.
- **Generalization:** ResNeXt performs well in various benchmark tasks, including object detection and semantic segmentation, demonstrating its generalizability across different domains of computer vision.