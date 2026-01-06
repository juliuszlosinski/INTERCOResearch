AlexNet is a deep convolutional neural network (CNN) architecture designed by Alex Krizhevsky in 2012. It won the **ImagNet Large Scale Visual Recognition Challange (ILSVRC 2012)** competition. The architecture consits of eight layers: **five convolutional** layers followed by **three fully connected** layers. It uses **ReLU (*Rectified Linear Unit*)** activation function, regularization **dropout** technique and **max pooling**. It was one of the first models that used GPU acceleration in order to make deep learning more practical and efficient. This success is often seen as the breakthrough that kicked off the modern **deep learning revolution** in computer vision (CV) area.

**Architecture Details**:
- **Input:** 224×224×3 RGB image
- **Layers:**
    - **5 Convolutional Layers** – extract features like edges, textures, and object parts.
    - **3 Fully Connected Layers** – perform final classification.
    - **Softmax Output Layer** – gives probabilities for 1000 object categories.
- **Activation function:**
    - Uses **ReLU (Rectified Linear Unit)** instead of sigmoid/tanh, which made training **much faster** by reducing vanishing gradient problems.
- **Pooling:**
    - **Max pooling** layers reduce spatial dimensions while preserving key features.
- **Regularization:**
    - **Dropout** was introduced in the fully connected layers to prevent overfitting by randomly disabling neurons during training.
- **Normalization:**
    - **Local Response Normalization (LRN)** was used to help generalization, though it's less common in newer models.
- **GPU acceleration:**
	- The network was trained on **two NVIDIA GTX 580 GPUs**, splitting layers across them - this a major innovation at the time and made large-scale deep learning feasible.

**Training details:**
- **Dataset:** ImageNet (1.2 million images, 1000 classes),
- **Data augmentation:** Random cropping, flipping and color jittering wer used to increase diversity and reduce overfitting,
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum,

**Impact:**
- Proved that **deep neural networks** could outperform traditional computer vision methods,
- Inspired later architectures like **VGGNet**, **GoogLeNet (Inception)** and **ResNet (Residual Network)**,
- Marked the start of the **deep learning era** in **artificial intelligence area**, influencing applications in vision, speech and natural language processing.

|**Layer**|**Type**|**Filter Size / Units**|**Stride / Padding**|**Activation**|**Output Size**|**Notes**|
|---|---|---|---|---|---|---|
|**Input**|Image|224×224×3|–|–|224×224×3|RGB image|
|**Conv1**|Convolution|11×11×3 filters, 96 kernels|Stride 4, Pad 0|ReLU|55×55×96|Large receptive field|
|**MaxPool1**|Max Pooling|3×3|Stride 2|–|27×27×96|Downsampling|
|**LRN1**|Local Response Normalization|–|–|–|27×27×96|Normalizes across channels|
|**Conv2**|Convolution|5×5×48 filters, 256 kernels|Stride 1, Pad 2|ReLU|27×27×256|Split across two GPUs|
|**MaxPool2**|Max Pooling|3×3|Stride 2|–|13×13×256|Downsampling|
|**LRN2**|Local Response Normalization|–|–|–|13×13×256|Normalization again|
|**Conv3**|Convolution|3×3×256 filters, 384 kernels|Stride 1, Pad 1|ReLU|13×13×384|Full connection of feature maps|
|**Conv4**|Convolution|3×3×192 filters, 384 kernels|Stride 1, Pad 1|ReLU|13×13×384|GPU-split layer|
|**Conv5**|Convolution|3×3×192 filters, 256 kernels|Stride 1, Pad 1|ReLU|13×13×256|Feature extraction|
|**MaxPool3**|Max Pooling|3×3|Stride 2|–|6×6×256|Final spatial reduction|
|**FC6**|Fully Connected|4096 neurons|–|ReLU + Dropout|1×1×4096|Flattened input from conv layers|
|**FC7**|Fully Connected|4096 neurons|–|ReLU + Dropout|1×1×4096|High-level features|
|**FC8**|Fully Connected|1000 neurons|–|Softmax|1×1×1000|Classification output (ImageNet classes)|

Input (224×224×3)
   ↓
Conv1 → ReLU → LRN → MaxPool
   ↓
Conv2 → ReLU → LRN → MaxPool
   ↓
Conv3 → ReLU
   ↓
Conv4 → ReLU
   ↓
Conv5 → ReLU → MaxPool
   ↓
Flatten → FC6 → Dropout → ReLU
   ↓
FC7 → Dropout → ReLU
   ↓
FC8 → Softmax

**Key notes:**
- **Depth:** 8 learned layers (5 convolutional + 3 fully connected),
- **Parameters:** ~60 million - very huge for its time (2012),
- **Training hardware:** Two NVIDIA GTX 580 GPUs (3GB each),
- **Training time:** About 5-6 days on ImageNet (2012),
- **Breakthrough:** Showed the power of deep CNNs + GPUs + ReLU + data augmentation.