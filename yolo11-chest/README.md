# 使用yolo11进行的肺结节检测（Luna2016）

🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳

视频地址：[手把手教你使用YOLO11实现车辆检测与追踪系统_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1nzzdYwE2g/)

🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳🥳

各位小伙伴，大家好，这里是肆十二，今天我给大家带了的是基于yolo11的肺结节检测系统，该系统使用tt100k交通识别数据进行开发，一共包含有42个类的肺结节，我们分别基于yolov5、yolov8和yolo11进行了训练。本博客中我们将会按照教会大家对这个数据集进行训练、测试以及使用图形化的界面进行模型的加载来完成图像和视频的检测，效果图如下所示。

![image-20241211204235380](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241211204235380.png)

![image-20241211204736357](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241211204736357.png)

![image-20241211204753525](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241211204753525.png)

## 项目实战

进行项目实战之前请务必安装好pytorch和miniconda。

不会的小伙伴请看这里：[Python项目配置前的准备工作-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/144233262?sharetype=blogdetail&sharerId=144233262&sharerefer=PC&sharesource=ECHOSON&spm=1011.2480.3001.8118)

### 环境配置

执行下列指令创建并激活虚拟环境

```bash
conda create -n yolo python==3.8.5
conda activate yolo
```

执行下列执行安装pytorch

```bash
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # 30系列以上显卡gpu版本pytorch安装指令
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 #服务器的小伙伴使用这个
```

在**项目目录下**执行下列指令进行其他库的安装

```bash
pip install -v -e .
```

环境创建完成之后请使用pycharm打开你的项目，并在pycharm的右下角选择你项目对应的虚拟环境。

### 本地模型训练

模型训练使用的脚本为` step1_start_train.py `，进行模型训练之前，请先按照配置好你本地的数据集。数据集在` ultralytics\cfg\datasets\A_my_data.yaml`目录下，你需要将数据集的根目录更换为你自己本地的目录。

![image-20241204100852481](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204100852481.png)

更换之后修改训练脚本配置文件的路径，直接右键即可开始训练。

![image-20241204100955839](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204100955839.png)

如果你想要在gpu上训练，请将这里的device设置为0

![image-20241204101046196](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101046196.png)

训练开始前如果出现报错，有很大的可能是数据集的路径没有配置正确，请检查数据集的路径，保证数据集配置没有问题。训练之后的结果将会保存在runs目录下。

![image-20241204101214326](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101214326.png)

### GPU服务器训练（可选）

目前蓝耘GPU可以薅羊毛，推荐小伙伴从这个网站使用GPU云来进行训练，新用户注册会获得30元的代金券。

注册地址：[蓝耘GPU智算云平台](https://cloud.lanyun.net/#/registerPage?promoterCode=5c9cd7436a)

服务器使用指南：[手把手教你使用服务器训练AI模型_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1TuxLeVED6?vd_source=2f9a4e63109c3db3be5e8078e5111776&spm_id_from=333.788.videopod.sections)

### 模型测试

模型的测试主要是对map、p、r等指标进行计算，使用的脚本为` step2_start_val.py`，模型在训练的最后一轮已经执行了测试，其实这个步骤完全可以跳过，但是有的朋友可能想要单独验证，那你只需要更改测试脚本中的权重为你自己所训练的权重路径，即可单独进行测试。

![image-20241204101429118](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101429118.png)

### 图形化界面封装

图形化界面进行了升级，本次图形化界面的开发我们使用pyside6来进行开发。**PySide6** 是一个开源的Python库，它是Qt 6框架的Python绑定。Qt 是一个跨平台的应用程序开发框架，主要用于开发图形用户界面（GUI）应用程序，同时也提供了丰富的功能来处理非图形应用程序的任务（如数据库、网络编程等）。PySide6 使得开发者能够使用 Python 编写 Qt 6 应用程序，因此，它提供了Python的灵活性和Qt 6的强大功能。图形化界面提供了图片和视频检测等多个功能，图形化界面的程序为` step3_start_window_track.py `。

如果你重新训练了模型，需要替换为你自己的模型，请在这里进行操作。

![image-20241204101842858](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101842858.png)

如果你想要对图形化界面的题目、logo等进行修改，直接在这里修改全局变量即可。

![image-20241204101949741](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241204101949741.png)

登录之后上传图像或者是上传视频进行检测即可。



![image-20241211204235380](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241211204235380.png)

![image-20241211204736357](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241211204736357.png)

![image-20241211204753525](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241211204753525.png)

## 文档

### 背景与意义

使用YOLOv11进行肺结节检测的背景和意义主要体现在以下几个方面：

1. **提高肺癌早期筛查的效率和准确性**：肺癌是全球发病率和死亡率最高的恶性肿瘤之一，早期发现肺结节对于肺癌的预防和治疗至关重要。YOLOv11作为一种深度学习模型，能够提高肺结节的检测率，尤其是在早期肺癌筛查中。

2. **减轻医生工作负担**：随着CT数据的增加，使用计算机辅助诊断（CAD）系统可以大大减少放射科医生的工作量，并降低漏诊率。

3. **提高检测的泛化能力和准确性**：YOLOv11模型通过结合图像增强、噪声过滤和数据扩充等预处理步骤，能够学习到肺结节的多种特征，从而提高模型的泛化能力和准确性。

4. **减少误诊和漏诊**：实验结果表明，YOLOv11在肺结节检测任务中具有较高的精度和召回率，能够有效减少假阳性和假阴性的发生。

5. **提高检测速度**：与传统的检测方法相比，YOLOv11在检测速度和精度上都具有明显的优势，为医学影像智能分析提供了切实可行的解决方案。

6. **辅助临床诊断**：YOLOv11不仅能够减轻医生的工作负担，还可以通过其高效性和可靠性为医疗诊断提供有力的技术支持。

7. **推动个性化医疗的发展**：通过肺结节的早期检测和诊断，YOLOv11有助于促进个性化医疗的发展，最终提高患者的生存率和生活质量。

8. **探索深度学习在医学影像分析中的应用**：YOLOv11的研究不仅响应了医学影像分析领域对高效、准确的自动化工具的迫切需求，也为深度学习技术在医学领域的应用探索提供了新的方向。

综上所述，使用YOLOv11进行肺结节检测具有重要的临床意义和研究价值，它不仅能够提高肺结节检测的效率和准确性，还能够辅助医生进行更准确的诊断，从而在肺癌的早期发现和治疗中发挥关键作用。

### 相关文献综述

随着CT数据的日益增长，使用计算机辅助诊断(CAD)系统可以极大减少放射科医生的工作量以及降低漏检率。CAD系统通过肺实质分割、候选结节检测以及假阳性剔除三个阶段来实现肺结节的高效检测。

肺实质分割是检测前的重要预处理步骤，其目的是将肺结节的检测范围限制在肺部区域。研究中提出了结合U-Net与传统图像处理方法的肺实质分割技术，以及结合ResNet和RCNN的分割模型RU-Net和R2U-Net，这些模型在肺实质分割数据集上取得了较高的Dice系数

早期的候选结节检测多采用阈值分割、形态学运算等传统方法，但这些方法难以提取到判别性特征，导致假阳性候选结节的生成。随着深度学习技术的发展，基于深度学习的候选结节检测方法，如结合3D RPN的方法，以及基于3D Faster RCNN的多任务端到端模型，显著提高了检测的灵敏度和准确性

尽管现有的肺结节检测方法取得了较为满意的效果，但仍存在挑战，如基于深度学习的方法依赖于大量高质量的金标准数据，而现有的胸部CT公开数据集并没有进行有序的标记，导致金标准数据稀缺。未来的研究需要进一步优化算法，提高检测的准确性，并解决数据标注和隐私保护的问题。

### 本文算法介绍

yolo系列已经在业界可谓是家喻户晓了，下面是yolo11放出的性能测试图，其中这种图的横轴为模型的速度，一般情况下模型的速度是通过调整卷积的深度和宽度来进行修改的，纵轴则表示模型的精度，可以看到在同样的速度下，11表现出更高的精度。

![image-20241024170914031](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024170914031.png)

YOLO架构的核心由三个基本组件组成。首先，主干作为主要特征提取器，利用卷积神经网络将原始图像数据转换成多尺度特征图。其次，颈部组件作为中间处理阶段，使用专门的层来聚合和增强不同尺度的特征表示。第三，头部分量作为预测机制，根据精细化的特征映射生成目标定位和分类的最终输出。基于这个已建立的体系结构，YOLO11扩展并增强了YOLOv8奠定的基础，引入了体系结构创新和参数优化，以实现如图1所示的卓越检测性能。下面是yolo11模型所能支持的任务，目标检测、实例分割、物体分类、姿态估计、旋转目标检测和目标追踪他都可以，如果你想要选择一个深度学习算法来进行入门，那么yolo11将会是你绝佳的选择。

![image-20241024171109729](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024171109729.png)

为了能够让大家对yolo11网络有比较清晰的理解，下面我将会对yolo11的结构进行拆解。

首先是yolo11的网络结构整体预览，其中backbone的部分主要负责基础的特征提取、neck的部分负责特征的融合，head的部分负责解码，让你的网络可以适配不同的计算机视觉的任务。

![image-20241024173654996](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024173654996.png)

* 主干网络（BackBone）

  * Conv

    卷积模块是一个常规的卷积模块，在yolo中使用的非常多，可以设计卷积的大小和步长，代码的详细实现如下：

    ```python
    class Conv(nn.Module):
        """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    
        default_act = nn.SiLU()  # default activation
    
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            """Initialize Conv layer with given arguments including activation."""
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
        def forward(self, x):
            """Apply convolution, batch normalization and activation to input tensor."""
            return self.act(self.bn(self.conv(x)))
    
        def forward_fuse(self, x):
            """Perform transposed convolution of 2D data."""
            return self.act(self.conv(x))
    ```

  * C3k2

    C3k2块被放置在头部的几个通道中，用于处理不同深度的多尺度特征。他的优势有两个方面。一个方面是这个模块提供了更快的处理:与单个大卷积相比，使用两个较小的卷积可以减少计算开销，从而更快地提取特征。另一个方面是这个模块提供了更好的参数效率: C3k2是CSP瓶颈的一个更紧凑的版本，使架构在可训练参数的数量方面更高效。

    C3k2模块主要是为了增加特征的多样性，其中这块模块是由C3k模块演变而来。它通过允许自定义内核大小提供了增强的灵活性。C3k的适应性对于从图像中提取更详细的特征特别有用，有助于提高检测精度。C3k的实现如下。

    ```python
    class C3k(C3):
        """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
    
        def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
            """Initializes the C3k module with specified channels, number of layers, and configurations."""
            super().__init__(c1, c2, n, shortcut, g, e)
            c_ = int(c2 * e)  # hidden channels
            # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
    ```

    如果将c3k中的n设置为2，则此时的模块即为C3K2模块，网络结构图如下所示。

    ![image-20241025121912923](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241025121912923.png)

    该网络的实现代码如下。

    ```python
    class C3k2(C2f):
        """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    
        def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
            """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
            super().__init__(c1, c2, n, shortcut, g, e)
            self.m = nn.ModuleList(
                C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
            )
    ```

  * C2PSA

    PSA的模块起初在YOLOv10中提出，通过自注意力的机制增加特征的表达能力，相对于传统的自注意力机制而言，计算量又相对较小。网络的结构图如下所示，其中图中的mhsa表示的是多头自注意力机制，FFN表示前馈神经网络。

    ![image-20241025122617233](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241025122617233.png)

    

  在这个基础上添加给原先的C2模块上添加一个PSA的旁路则构成了C2PSA的模块，该模块的示意图如下。

  ![image-20241025122752167](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241025122752167.png)

  网络实现如下：

  ```python
  class C2PSA(nn.Module):
      """
      C2PSA module with attention mechanism for enhanced feature extraction and processing.
  
      This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
      capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
  
      Attributes:
          c (int): Number of hidden channels.
          cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
          cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
          m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.
  
      Methods:
          forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.
  
      Notes:
          This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
  
      Examples:
          >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
          >>> input_tensor = torch.randn(1, 256, 64, 64)
          >>> output_tensor = c2psa(input_tensor)
      """
  
      def __init__(self, c1, c2, n=1, e=0.5):
          """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
          super().__init__()
          assert c1 == c2
          self.c = int(c1 * e)
          self.cv1 = Conv(c1, 2 * self.c, 1, 1)
          self.cv2 = Conv(2 * self.c, c1, 1)
  
          self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
  
      def forward(self, x):
          """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
          a, b = self.cv1(x).split((self.c, self.c), dim=1)
          b = self.m(b)
          return self.cv2(torch.cat((a, b), 1))
  
  ```

* 颈部网络（Neck）

  * upsample

    这里是一个常用的上采样的方式，在YOLO11的模型中，这里一般使用最近邻差值的方式来进行实现。在 `torch`（PyTorch）中，`upsample` 操作是用于对张量（通常是图像或特征图）进行**上采样**（增大尺寸）的操作。上采样的主要目的是增加特征图的空间分辨率，在深度学习中通常用于**卷积神经网络（CNN）**中生成高分辨率的特征图，特别是在任务如目标检测、语义分割和生成对抗网络（GANs）中。

    PyTorch 中的 `torch.nn.functional.upsample` 在较早版本提供了上采样功能，但在新的版本中推荐使用 `torch.nn.functional.interpolate`，功能相同，但更加灵活和标准化。

    主要参数如下：

    `torch.nn.functional.interpolate` 函数用于上采样，支持不同的插值方法，常用的参数如下：

    ```python
    torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
    ```

    - `input`：输入的张量，通常是 4D 的张量，形状为 `(batch_size, channels, height, width)`。

    - `size`：输出的目标尺寸，可以是整型的高度和宽度（如 `(height, width)`），表示希望将特征图调整到的具体尺寸。

    - `scale_factor`：上采样的缩放因子。例如，`scale_factor=2` 表示特征图的高度和宽度都扩大 2 倍。如果设置了 `scale_factor`，则不需要再设置 `size`。

    - ```
      mode
      ```

      ：插值的方式，有多种可选插值算法：

      - `'nearest'`：最近邻插值（默认）。直接复制最近的像素值，计算简单，速度快，但生成图像可能比较粗糙。
      - `'linear'`：双线性插值，适用于 3D 输入（即 1D 特征图）。
      - `'bilinear'`：双线性插值，适用于 4D 输入（即 2D 特征图）。
      - `'trilinear'`：三线性插值，适用于 5D 输入（即 3D 特征图）。
      - `'bicubic'`：双三次插值，计算更复杂，但生成的图像更平滑。

    - `align_corners`：在使用双线性、三线性等插值时决定是否对齐角点。如果为 `True`，输入和输出特征图的角点会对齐，通常会使插值结果更加精确。

  * Concat

    在YOLO（You Only Look Once）目标检测网络中，`concat`（连接）操作是用于将来自不同层的特征图拼接起来的操作。其作用是融合不同尺度的特征信息，以便网络能够在多个尺度上更好地进行目标检测。调整好尺寸后，沿着**通道维度**将特征图进行拼接。假设我们有两个特征图，分别具有形状 (H, W, C1) 和 (H, W, C2)，拼接后得到的特征图形状将是 (H, W, C1+C2)，即通道数增加了。一般情况下，在进行concat操作之后会再进行一次卷积的操作，通过卷积的操作可以将通道数调整到理想的大小。该操作的实现如下。

    ```python
    class Concat(nn.Module):
        """Concatenate a list of tensors along dimension."""
    
        def __init__(self, dimension=1):
            """Concatenates a list of tensors along a specified dimension."""
            super().__init__()
            self.d = dimension
    
        def forward(self, x):
            """Forward pass for the YOLOv8 mask Proto module."""
            return torch.cat(x, self.d)
    ```

* 头部（Head）

  YOLOv11的Head负责生成目标检测和分类方面的最终预测。它处理从颈部传递的特征映射，最终输出图像内对象的边界框和类标签。一般负责将特征进行映射到你对应的任务上，如果是检测任务，对应的就是4个边界框的值以及1个置信度的值和一个物体类别的值。如下所示。

  ```python
  # Ultralytics YOLO 🚀, AGPL-3.0 license
  """Model head modules."""
  
  import copy
  import math
  
  import torch
  import torch.nn as nn
  from torch.nn.init import constant_, xavier_uniform_
  
  from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
  
  from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
  from .conv import Conv, DWConv
  from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
  from .utils import bias_init_with_prob, linear_init
  
  __all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect"
  
  
  ```

基于上面的设计，yolo11衍生出了多种变种，如下表所示。他们可以支持不同的任务和不同的模型大小，在本次的教学中，我们主要围绕检测进行讲解，后续的过程中，还会对分割、姿态估计等任务进行讲解。

![image-20241024173356022](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241024173356022.png)

YOLOv11代表了CV领域的重大进步，提供了增强性能和多功能性的引人注目的组合。YOLO架构的最新迭代在精度和处理速度方面有了显著的改进，同时减少了所需参数的数量。这样的优化使得YOLOv11特别适合广泛的应用程序，从边缘计算到基于云的分析。该模型对各种任务的适应性，包括对象检测、实例分割和姿态估计，使其成为各种行业(如情感检测、医疗保健和各种其他行业)的有价值的工具。它的无缝集成能力和提高的效率使其成为寻求实施或升级其CV系统的企业的一个有吸引力的选择。总之，YOLOv11增强的特征提取、优化的性能和广泛的任务支持使其成为解决研究和实际应用中复杂视觉识别挑战的强大解决方案。

### 实验结果分析

#### 数据集介绍

本次我们使用的数据集为LUNA 2016 ：[数据集官方地址]([Home - LUNA16 - Grand Challenge](https://luna16.grand-challenge.org/))
LUNA 2016 数据集来自2016年LUng Nodule Analysis比赛，这里是其官方网站。
LUNA16数据集是最大公用肺结节数据集LIDC-IDRI的子集，LIDC-IDRI它包括1018个低剂量的肺部CT影像。LIDC-IDRI删除了切片厚度大于3mm和肺结节小于3mm的CT影像，剩下的就是LUNA16数据集了。
其数据集的文件内容如下：

1. subset0.zip~subset9.zip 包含所有CT图像的10个zip文件，数据格式为“.mhd”,“.raw”
2. CSVFILES文件夹，包括3文件：annotations.csv，candidates.csv，sampleSubmission.csv
3. annotations.csv：1186个肺结节信息，字段有seriesuid,coordX,coordY,coordZ,diameter_mm
4. candidates.csv：一共551065条数据。其中，正例（class：1）：1351条，其余都是负例（class：0）
5. sampleSubmission.csv：正确格式的提交文件示例，我们不提交，暂时没用。

**mdh数据格式详解**
这里有一篇很好的文章，关于医疗影像的mhd和dcm格式图像的读取和坐标转换
每个病例的数据的存储都是由一个.mhd和一个.raw格式的文件组成。.mdh是说明文件，具体数据在.raw文件中，mdh的样例如下：

```
ObjectType = Image
NDims = 3          #三维数据
BinaryData = True              #二进制数据
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1        #100,010,001 分别代表x,y,z
Offset = -195 -195 -378       #原点坐标
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = 0.7617189884185791 0.7617189884185791 2.5     #像素间隔 x,y,z
DimSize = 512 512 141        #数据的大小 x,y,z
ElementType = MET_SHORT
ElementDataFile = 1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845.raw      #数据存储的文件名

```

我们可以使用python中的simpleITK库对mdh文件进行解析，代码如下：

```python
import SimpleITK as sitk
import matplotlib.pyplot as plt
case_path = './1.3.6.1.4.1.14519.5.2.1.6279.6001.126264578931778258890371755354.mhd'  
itkimage = sitk.ReadImage(case_path)   #这部分给出了关于图像的信息,可以打印处理查看，这里就不在显示了
#print(itkimage)
image = sitk.GetArrayFromImage(itkimage)     #z,y,x
#查看第100张图像
plt.figure()
plt.imshow(image[100,:,:]) 

```

另外，为了获得肺结节的位置信息，我们需要对annotations.csv文件进行作为坐标转化。

annotations.csv中提供了医生标注肺结节位置信息
seriesuid：表示每个病例图像对应的文件名
coordX，coordX，coordX，diameter_mm:表示医生标注的结节位置信息和直径

![img](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/e0b47bf813a1b6b586730fa25cd047f6.png)

在使coordX用卷积网络对肺结节进行检测时，我们需要根据医生提供的标注信息，在图像中找到相应的肺结节位置，接下来说医生标注的坐标与图像中的坐标的关系。
mhd中给定了图像中的原点坐标为（-195 ，-195 ，-378） #x,y,z
像素间隔为（0.7617189884185791，0.7617189884185791，2.5） #x,y,z
通过以上信息可以计算结节相对原点的坐标，然后用这个坐标除以像素间隔，即为在图像中对应的结节位置
#世界坐标转换到图像中的坐标，函数如下：

```python
def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord
```

图像上的坐标转化为世界坐标的脚本如下

```python
def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord
```

让我们一起可视化看下效果。

```python
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
filename='data\\1.3.6.1.4.1.14519.5.2.1.6279.6001.173106154739244262091404659845.mhd'
itkimage = sitk.ReadImage(filename)#读取.mhd文件
OR=itkimage.GetOrigin()
print(OR)
SP=itkimage.GetSpacing()
print(SP)
numpyImage = sitk.GetArrayFromImage(itkimage)#获取数据，自动从同名的.raw文件读取

def show_nodules(ct_scan, nodules,Origin,Spacing,radius=20, pad=2, max_show_num=4): # radius是正方形边长一半，pad是边的宽度,max_show_num最大展示数
    show_index = []
    for idx in range(nodules.shape[0]): # lable是一个nx4维的数组，n是肺结节数目，4代表x,y,z,以及直径
        if idx < max_show_num:
            if abs(nodules[idx, 0]) + abs(nodules[idx, 1]) + abs(nodules[idx, 2]) + abs(nodules[idx, 3]) == 0:
                continue

            x, y, z = int((nodules[idx, 0]-Origin[0])/SP[0]), int((nodules[idx, 1]-Origin[1])/SP[1]), int((nodules[idx, 2]-Origin[2])/SP[2])
        print(x, y, z)
        data = ct_scan[z]
        radius=int(nodules[idx, 3]/SP[0]/2)
        #pad = 2*radius
        # 注意 y代表纵轴，x代表横轴
        data[max(0, y - radius):min(data.shape[0], y + radius),
        max(0, x - radius - pad):max(0, x - radius)] = 3000 # 竖线
        data[max(0, y - radius):min(data.shape[0], y + radius),
        min(data.shape[1], x + radius):min(data.shape[1], x + radius + pad)] = 3000 # 竖线
        data[max(0, y - radius - pad):max(0, y - radius),
        max(0, x - radius):min(data.shape[1], x + radius)] = 3000 # 横线
        data[min(data.shape[0], y + radius):min(data.shape[0], y + radius + pad),
        max(0, x - radius):min(data.shape[1], x + radius)] = 3000 # 横线

        if z in show_index: # 检查是否有结节在同一张切片，如果有，只显示一张
            continue
        show_index.append(z)
        plt.figure(idx)
        plt.imshow(data, cmap='gray')

    plt.show()

b = np.array([[-116.2874457,21.16102581,-124.619925,10.88839157],[-111.1930507,-1.264504521,-138.6984478,17.39699158],[73.77454834,37.27831567,-118.3077904,8.648347161]])
show_nodules(numpyImage,b,OR,SP)
```

可视化之后的结果如下所示

![image-20241212100559083](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/image-20241212100559083.png)

我在这里已经将肺结节的数据按照yolo格式进行了处理，大家只需要在配置文件种对本地的数据地址进行配置即可，如下所示。

```yaml
path: /root/autodl-tmp/yolo-data/tt100/reTT100K
train: # train images (relative to 'path')  16551 images
  - images/train
val: # val images (relative to 'path')  4952 images
  - images/val
test: # test images (optional)
  - images/test

names: ['i2','i4','i5','il100','i160','il80','io','ip','p10','p11',
 'p12','p19','p23','p26','p27','p3','p5','pó','pg','ph4','ph4.5',
 'pl100','pl120','pl20','pl30','pl40','pl5','pl50','pl60','pl70',
 'pL80','pm20','pm30','pm55','pn','pne','po','pr40','w13','w55',
 'w57', 'w59']
```

下面是数据集的部分示例。

![val_batch0_labels](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/val_batch0_labels.jpg)

以及下面是数据集中每个类别对应的实例数量和边界框大小的基本分析，从下图可以看出，大部分目标都比较小，属于是小目标检测的内容。

![labels - chest](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/labels%20-%20chest.jpg)

#### 实验结果分析

实验结果的指标图均保存在runs目录下， 大家只需要对实验过程和指标图的结果进行解析即可。

如果只指标图的定义不清晰，请看这个位置：[YOLO11模型指标解读-mAP、Precision、Recall_yolo11模型训练特征图-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/144097341)

![results-chest](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/results-chest.png)

train/box_loss（训练集的边界框损失）：随着训练轮次的增加，边界框损失逐渐降低，表明模型在学习更准确地定位目标。
train/cls_loss（训练集的分类损失）：分类损失在初期迅速下降，然后趋于平稳，说明模型在训练过程中逐渐提高了对肺结节的分类准确性。
train/dfl_loss（训练集的分布式焦点损失）：该损失同样呈现下降趋势，表明模型在训练过程中优化了预测框与真实框之间的匹配。
metrics/precision(B)（精确度）：精确度随着训练轮次的增加而提高，说明模型在减少误报方面表现越来越好。
metrics/recall(B)（召回率）：召回率也在逐渐上升，表明模型能够识别出更多的真实肺结节。
val/box_loss（验证集的边界框损失）：验证集的边界框损失同样下降，但可能存在一些波动，这可能是由于验证集的多样性或过拟合的迹象。
val/cls_loss（验证集的分类损失）：验证集的分类损失下降趋势与训练集相似，但可能在某些点上出现波动。
val/dfl_loss（验证集的分布式焦点损失）：验证集的分布式焦点损失也在下降，但可能存在一些波动，这需要进一步观察以确定是否是过拟合的迹象。
metrics/mAP50(B)（在IoU阈值为0.5时的平均精度）：mAP50随着训练轮次的增加而提高，表明模型在检测任务上的整体性能在提升。
metrics/mAP50-95(B)（在IoU阈值从0.5到0.95的平均精度）：mAP50-95的提高表明模型在不同IoU阈值下的性能都在提升，这是一个更严格的性能指标。

<img src="https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/PR_curve-chest.png" alt="PR_curve-chest" style="zoom:29%;" />

当iou阈值为0.5的时候，模型在测试集上的map可以达到68.1%。下面是一个预测图像，可以看出，我们的模型可以有效的预测出这些尺度比较小的交通目标。

![val_batch1_pred-chest](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/imgs/val_batch1_pred-chest.jpg)

### 结论

综上所述，基于YOLOv11的肺结节检测系统为医学影像分析领域提供了一种高效、准确的解决方案，有望在临床实践中得到广泛应用，从而提高肺癌的早期诊断率和治疗成功率。

### 参考文献

[1] 王德杭,汪家旺,于立燕,等.肺结节检测算法研究[J].生物医学工程学进展, 2005, 26(001):3-7.DOI:10.3969/j.issn.1674-1242.2005.01.001.

[2] 高园园,吕庆文,郭宏,等.一种新的肺结节检测算法[J].计算机工程与应用, 2007, 43(23):2.DOI:10.3321/j.issn:1002-8331.2007.23.060.

[3] 李丽.基于CT影像分析的肺结节检测算法研究[D].大连理工大学,2010.DOI:CNKI:CDMD:2.1011.023110.

[4]   Zhou Q , Yu C . Point RCNN: An Angle-Free Framework for Rotated Object Detection[J]. Remote Sensing, 2022, 14.

[5]  Zhang, Y., Li, H., Bu, R., Song, C., Li, T., Kang, Y., & Chen, T. (2020). Fuzzy Multi-objective Requirements for NRP Based on Particle Swarm Optimization. *International Conference on Adaptive and Intelligent Systems*.

[6]   Li X , Deng J , Fang Y . Few-Shot Object Detection on Remote Sensing Images[J]. IEEE Transactions on Geoscience and Remote Sensing, 2021(99).

[7]   Su W, Zhu X, Tao C, et al. Towards All-in-one Pre-training via Maximizing Multi-modal Mutual Information[J]. arXiv preprint arXiv:2211.09807, 2022.

[8]   Chen Q, Wang J, Han C, et al. Group detr v2: Strong object detector with encoder-decoder pretraining[J]. arXiv preprint arXiv:2211.03594, 2022.

[9]   Liu, Shilong, et al. "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection." arXiv preprint arXiv:2303.05499 (2023).

[10] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

[11] Redmon J, Farhadi A. YOLO9000: better, faster, stronger[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7263-7271.

[12] Redmon J, Farhadi A. Yolov3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.

[13] Tian Z, Shen C, Chen H, et al. Fcos: Fully convolutional one-stage object detection[C]//Proceedings of the IEEE/CVF international conference on computer vision. 2019: 9627-9636.

[14] Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 801-818.

[15] Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector[C]//Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14. Springer International Publishing, 2016: 21-37.

[16] Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 2117-2125.

[17] Cai Z, Vasconcelos N. Cascade r-cnn: Delving into high quality object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 6154-6162.

[18] Ren S, He K, Girshick R, et al. Faster r-cnn: Towards real-time object detection with region proposal networks[J]. Advances in neural information processing systems, 2015, 28.

[19] Wang R, Shivanna R, Cheng D, et al. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems[C]//Proceedings of the web conference 2021. 2021: 1785-1797.

[20] Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.
