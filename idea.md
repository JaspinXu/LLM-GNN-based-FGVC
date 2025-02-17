# Fine grained Classification

细粒度分类

一、Weakly Supervised Posture Mining for Fine-grained Classification

cvpr 2023

问题：什么是细粒度分类？与平常的分类任务有啥区别？

答：fine grained classification是指细粒度分类，普通分类可以说是区分是猫还是狗，但是细粒度的分类是指我想去区分狗里面不同的类别是什么，传统的细粒度分类方法通常需要额外的详细标记信息，如每个物体的边界或特征部分（如鸟的头部、翅膀等）的位置，才能识别微小的差异，同类别之间的差异很小

![截屏2024-11-20 11.44.56.png](%25E6%2588%25AA%25E5%25B1%258F2024-11-20_11.44.56.png)

deep navigator：生成物体的突出区域。

基于图的分类：识别区域后，PMRC 构建一个图结构，其中节点代表突出区域，边代表它们之间的空间距离（RBF）。该图结构通过消息传递进行处理，整合每个区域及其空间关系的信息，从而实现更准确的分类。

二：Navigating Real-World Partial Label Learning: Unveiling Fine-Grained Images with Attributes

AAAI 2024

Method：提出了PLL-FG框架，主要包括三个模块：attribute space visual representation module, attribute attention mechanism, and dynamic disambiguation module.

![截屏2024-11-20 16.58.38.png](%25E6%2588%25AA%25E5%25B1%258F2024-11-20_16.58.38.png)

pipeline的理解：

1.图像经过backbone后得到特征图F,其中Fij是指的每一个像素点的特征

2.使用属性原型矩阵P映射属性图到特征空间，这里的p应该是由大语言模型得到的！F被映射到属性空间得到Fatt表示改区域与每个属性的匹配程度，然后，对整个图像的属性分布进行全局平均池化（GAP），生成全局的属性表示 Gatt，用于共享属性监督。在候选标签集中，我们将使用这些标签的共享属性来进行计算损失

3.针对候选标签的不同属性，模型将引导注意力去关注图像的关键区域，属性注意力机制通过计算注意力图 E，强化图像中与关键属性相关的区域特征

4.动态消歧模块：我们使用候选标签的置信度更新候选标签

5.训练的总损失函数结合了多个部分：

- Ltotal=Lcls+αLatt+βLce
- Lcls​：基于分类结果的损失，用于全局特征优化。
- Latt​：共享属性的监督损失，用于学习候选类别的共同特性。
- Lce​：简单样本的分类损失。

三、African or European Swallow? Benchmarking Large Vision-Language Models for Fine-Grained Object Classification

![截屏2024-11-20 17.33.26.png](%25E6%2588%25AA%25E5%25B1%258F2024-11-20_17.33.26.png)

四、Transitivity Recovering Decompositions: Interpretable and Robust Fine-Grained Relationships

2024 NIPS

![截屏2024-11-25 16.07.42.png](%25E6%2588%25AA%25E5%25B1%258F2024-11-25_16.07.42.png)

通过将其表达为图像视图上的可解释图形来解构这种抽象，设计了传递性恢复分解（TRD），这是一种图空间搜索算法。

类似于推理过

![截屏2024-11-25 16.46.45.png](%25E6%2588%25AA%25E5%25B1%258F2024-11-25_16.46.45.png)

五、Improved Zero-Shot Classification by Adapting VLMs with Text Descriptions

2024 CVPR

![image.png](image.png)

未微调前的问题：
原始的 VLM 是在大规模、通用的图像-文本对上训练的（如 ImageNet 和互联网上的图片），其特征提取器更擅长区分高层次类别（如“鸟 vs 狗”），而对于细粒度类别（如“红喉蜂鸟 vs 绿喉蜂鸟”）的区分能力有限。这主要是因为原始训练数据中缺乏细粒度特征的监督。
微调后的改进：
微调过程中，通过类别级别的文本描述（如红喉蜂鸟的“红色腹部、针状喙”），模型学会了更关注这些细粒度属性。图像编码器在提取特征时，开始捕捉到对细粒度分类有用的局部特征（如颜色、纹理、形状等）。

六、HARNESSING EXPLANATIONS: LLM-TO-LM INTERPRETER FOR ENHANCED TEXT-ATTRIBUTED GRAPH  REPRESENTATION LEARNING

文本属性图：节点往往是文本信息，如论文，节点的链接往往是论文之间的关系！！

![image.png](image%201.png)

七、Efficient Tuning and Inference for Large Language Models on Textual Graphs

![image.png](image%202.png)

总结一下：

目标marine science，先做海洋数据的Fine grained Classification

11.25第一次组会总结：
大致的idea是FGC和TAG的结合。目前的方法大致是通过考虑局部和全局的特征来进行细细粒度的分类！TAG这边有一些工作是利用大模型来进行更好的处理！我们希望利用大模型对局部和全局的特征进行总结

11.29
TAG有一篇工作是讲文章的摘要等给到大模型，然后利用大模型来判断这篇文章最应该属于哪一类的文章，并且要给出判据。接着把文章信息，属于哪类文章，判据都作为输入去微调大模型！利用调整好的大模型去做TAG的embedding！

我们的idea是利用TAG这边的做法去提高Fine grained Classification的性能！因为我们认为像TAG这种都是利用大模型去引入更多的信息！Fine grained Classification就需要更多的局部和全局的信息！是不是也可以利用llm的功能呢？例如：用llm去为每一张图像（局部or全局）去做描述！引入了一个额外的文字信息！

11.30

TAG工作的调研

1. **Large Language Model-based Augmentation for
Imbalanced Node Classification on Text-Attributed Graphs**

这篇文章提出了一个名为LA-TAG的新方法，利用大型语言模型（LLMs）的文本生成能力来处理TAG上的不平衡节点分类。具体来说，LA-TAG通过提示LLMs根据图形中现有的节点文本生成合成文本，来解决不平衡节点分类问题。此外，为了将这些合成的文本属性节点整合到图形中，文章引入了一个基于文本的链接预测器，将合成的节点与现有节点连接起来。

![截屏2024-11-30 22.01.44.png](%25E6%2588%25AA%25E5%25B1%258F2024-11-30_22.01.44.png)

SMOTE是一种数据增强方法，它通过在少数类别的节点周围合成新的样本来增加少数类别的数量（Chawla et al. 2002）。在LLM基础上，我们提出了一个改进的SMOTE方法，它首先找到原始节点在深度嵌入空间中的k个最近邻居，并确保它们属于同一类别。然后，它将这些邻居与原始节点的文本进行合成，以生成新的文本属性。

1. **LATEX-GCL: Large Language Models (LLMs)-Based Data Augmentation for Text-Attributed Graph Contrastive Learning**

关注文本增强的模块

![截屏2024-12-02 22.24.24.png](%25E6%2588%25AA%25E5%25B1%258F2024-12-02_22.24.24.png)

1.marine science！

2.利用VLM/LLM去给image引入更多的信息——————提升fine grained classification!

3.提高VLM的一个细粒度！

12.2

下周的任务调研数据集！

先用传统的网络跑！

然后调研sota的方法！

然后简单的用大模型去搭建一个网络！

然后看看我们方法的效果！！

调研任务

1数据集CUB-200-2011 , Stanford Cars , FGVC Aircraft , Stanford Dogs 

baseline:传统网络+PMRC+TRD+Relational Proxies+transfg

1

[https://github.com/ZhenchaoTang/Fine-grainedImageClassification](https://github.com/ZhenchaoTang/Fine-grainedImageClassification)

2

[https://github.com/abhrac/trd](https://github.com/abhrac/trd)

3

[https://github.com/abhrac/relational-proxies](https://github.com/abhrac/relational-proxies)

4

[https://github.com/TACJu/TransFG](https://github.com/TACJu/TransFG)

1.5

先搭建一下自己的pipeline吧！

目前来说先不用vlm！

这一块存在的问题：1.vlm的API难搞！ 2.运行速度感觉很慢！！

那就先用clip吧！

然后，这里的话打算先设置细粒度较低的concept set！！

用spice的方法去为局部和全局打上concept label作为text的额外信息embedding，同image embedding一起输入到graph里面！！

数据处理的差不多了，下面是用clip和字典来为patch image打标签，然后是产生图，然后是图分类！

这一周的任务：把大模型的部分写到pipeline里面，并且跑一跑代码，调一下参数！看看效果如何！

对不同的图像进行切割后，生成node的信息，然后里面有很多冗余的信息（这里可以考虑概念的提取！），我们可以使用信息密度来选择代表性的节点来排除无效的信息！

还有一个点就是prompt的设计！

具体开始实验：

目前的效果是68左右，效果还差不少！

后面的做法：1.backbone 把clip改成预训练的resnet，但是这有一个问题是他的维度和clip的维度不一样，怎么考虑多模态的引入！

1. backbone改为提高细粒度的clip！

试了一下api的描述，大致效果是全局描述很不错，局部描述很差！

实时的描述很难，我们提前对全局进行描述！存入json文件里面！这里可以实现鸟类的全局描述！

下一步怎么实现鸟类的局部描述！

其实可以直接生成，然后做两个图，去融合！

预设好concepts而抛弃细的细粒度描述！concepts怎么弄！

突然看到的paper，有了新的idea，就是我们想做一个信息增强的方式，可以无痛迁移到目前所存在的框架中！

**Free Lunch in Pathology Foundation Model: Task-specific Model Adaptation with Concept-Guided Feature Enhancement**

![截屏2024-12-17 15.45.28.png](%25E6%2588%25AA%25E5%25B1%258F2024-12-17_15.45.28.png)

学习的点：1.不需要用细的细粒度VLM来做信息增强！而是用concept set即可！

2.参考设计的CATE 的模块，做一个新的模块集成到fine grained classification的benchmark里面！

下面去跑其他的pipeline，然后，看看效果，设计好自己的信息增强模块！

细节：
prompt：What characteristics can be used to differentiate [class] from other [domain] based on just a photo? Provide an exhaustive list of all attributes that can be used to identify the [domain] uniquely. Texts should be of the form “[domain] with [characteristic]”.

===》不同class具体的概念！！

目前进度（1.20）

向trd作者发邮件询问，让在github的issue里提，杳无音讯

实际效果如下图，怀疑是训练集能到92%左右，测试集按默认参数跑完最多72%左右

![image.png](image%203.png)

但是框架很清晰，可以加很多东西

目前pipeline：

![865dac666ad8bdd1a4561ae75c40419.jpg](865dac666ad8bdd1a4561ae75c40419.jpg)

借鉴了CATE的思路改进，跑CUB的效果大致能提高到82和83左右

问题：llm根据prompt生成语言描述非常依赖数据集，每个数据集都要单独生成concept set和整体描述

需要进一步改进，

1.clip换成longclip，加强对长文本长序列的编码能力

2.调研新的增加信息的方式

前面的图将全局图片划分为局部子图，局部图之间互不重叠，然后侧重与局部图的语义扩展和局部图之间的联系，合理利用了节点之间的互信息，或许易受图片环境信息影响，捕捉到不相关信息

可不可以对每个局部图计算信息熵，然后求出根据信息熵值求出全局图的质心坐标即信息密度最大的位置，然后以其为中心画不同大小的圆，计算逐级之间的信息增益information gain，作为额外的信息补充到原来的图中，难点在于圆半径步长的界定以及加入方式(引入新的损失或者生成一个新的单节点增益图)

1.25目前my-pipeline跑出来的结果，35个epoch，训练集90%，测试集82%

![image.png](image%204.png)

my-method

![image.png](image%205.png)

PMRC:

![image.png](ea1168d7-b1d8-4257-a013-eacecb347f23.png)

TransFG:

![fcb7f8322ce5f1e159f1ffacd7051b3.png](fcb7f8322ce5f1e159f1ffacd7051b3.png)

**1.trd可以实现92%的acc**

![image.png](image%206.png)

![image.png](image%207.png)

### **2.加入longclip**

![image.png](image%208.png)

![image.png](image%209.png)

![image.png](image%2010.png)

![image.png](image%2011.png)

10个epoch82%→87%，30个epoch能到88.56%

### 3.实践了一下自己的信息增益方式

Q：为何选择环形区域而非扇形分区？

A：考虑旋转不变性需求，鸟类姿态变化较大，圆形对称特征更稳定

出现的问题：

①该过程需要频繁计算局部图坐标(特别是对于random抽取子图方式)，耗时增加

②本质上还是对于图像本身特征的提取，resnet特征提取能力已经足够

③应该侧重于TAG，利用大模型能力(比如最近火的deepseek R1)而非图像特征

如何进行数据集比较，LLM引入了descriptions，数据集变成image+label+description

引入llm增强语义如果训练下去完全能够超过TRD

要和TAG数据集比较(cora，pubmed，ogbn)还是图像数据集(CUB，FGVC等)

![image.png](image%2012.png)

2.5

加了concept_contrastive_loss进行对比学习，优化了处理逻辑

![image.png](image%2013.png)

结果10个epoch87%→87.46%，在25个epoch处就超过了之前的88.56%，到了近90%

但是后续基本稳定在这个值

加了第四个concept维度

2.8画了一下pipeline，但实际上还要加一些东西

![image.png](image%2014.png)

2.9

**Efficient Text-Attributed Graph Learning through Selective Annotation and Graph Alignment**

不同之处在于对于图像而言，子节点并没有像纯TAG领域的工作那样数量多，分的过小会导致误差变大，每个子图的信息不完整，分的过多会导致很多子图的相似性很高，产生信息冗余

下面是将子节点数量从32→64的变化，一个epoch从16→22分钟，准确率反而会降低87→86%左右

![image.png](image%2015.png)

对于图像而言的TAG也不用LM_encoder，Longclip就够用了，因为节点语义没有单个文献那样长

加了概念对比损失，但是不同之处在于是信息密度最高的局部图与正概念和负概念之间的对比损失

GCN

![image.png](image%2016.png)

GAT

![image.png](image%2017.png)

构建TAG使用GCN和GAT(注意力头设置为8)几乎差不多，反而是对子节点数量变化比较敏感

下一步计划

1.调用openai的api生成Stanford Cars , FGVC Aircraft数据集的descriptions和concept set

2.优化一下代码结构，目前一个epoch需要15-20分钟，不够快，速度提上来就可以开始跑实验

3.调一调参(划分的子节点数量，选取的概念数量等)，之前默认local = 32，concept = 20，在30个epoch左右时达到91%左右，稍微调一调参应该就能超过TRD了

4.从GNN结构、加与不加对比损失的差异、子节点和选取概念的数量的影响等角度设计一下消融实验

2.11

e3

```python
self.num_loacal  = 20
self.select_concepts  = 30
self.select_locals = 15
```

![image.png](image%2018.png)

e4

self.recovery_weight = 0.0001

![image.png](image%2019.png)

2.12

e5 原来数据集

![image.png](image%2020.png)

e6 常规数据集分割方法

![image.png](image%2021.png)

优化了代码处理逻辑

正在生成aircraft的descriptions

2.13

e7 常规数据集

```python
self.recovery_weight = 1
self.num_loacal  = 36
self.select_concepts  = 40
self.select_locals = 20
```

![image.png](image%2022.png)

2.14

优化了数据集预处理

aircraft

![image.png](image%2023.png)

CUB

![image.png](image%2024.png)

Soy和Cotton数据集，病虫害领域的叶片数据集

在生成concept set阶段就不适用于这类数据集

![image.png](image%2025.png)

Fish

[https://openaccess.thecvf.com/content/ICCV2023/papers/Khan_FishNet_A_Large-scale_Dataset_and_Benchmark_for_Fish_Recognition_Detection_ICCV_2023_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Khan_FishNet_A_Large-scale_Dataset_and_Benchmark_for_Fish_Recognition_Detection_ICCV_2023_paper.pdf)

[https://ieeexplore.ieee.org/abstract/document/9211789/](https://ieeexplore.ieee.org/abstract/document/9211789/)

Mammal

[https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_MammalNet_A_Large-Scale_Video_Benchmark_for_Mammal_Recognition_and_Behavior_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_MammalNet_A_Large-Scale_Video_Benchmark_for_Mammal_Recognition_and_Behavior_CVPR_2023_paper.pdf)

2.16

opt

```python
self.optimizer = Adam(trainable_params, lr=1e-3)  
self.scheduler = ReduceLROnPlateau(self.optimizer,mode='min',factor=0.1,patience=3,
verbose=True,min_lr=1e-6)
```

![image.png](image%2026.png)

bs16→32

![image.png](image%2027.png)

原来的dataset划分方式+resnet微调

![image.png](image%2028.png)

现在dataset划分方式+resnet微调

![image.png](image%2029.png)

old方法

![image.png](image%2030.png)

更新resnet方法

![image.png](image%2031.png)

2.17

olddata

![image.png](image%2032.png)

preold

![image.png](image%2033.png)

[CUB-200-2011 Benchmark (Fine-Grained Image Classification) | Papers With Code](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=metaformer-a-unified-meta-framework-for-fine)