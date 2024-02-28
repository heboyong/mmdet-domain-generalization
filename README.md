
## 简介

mmdet-domain-adaption 是一个基于[OpenMMLab](https://openmmlab.com/) 和 [MMDetection](https://github.com/open-mmlab/mmdetection/) 的项目，主要用于域自适应目标检测算法的开发和应用。

目前代码分支基于MMDetection-3.x版本。


### 1. 安装流程
(1) 参考  [MMDetection](https://github.com/open-mmlab/mmdetection/) 3.x分支安装流程，安装环境和依赖包。
目前版本：mmdet-3.0rc6

(2) 克隆本项目到本地


### 2. 主要内容
<details open>

- **有监督域自适应**

- **基于对抗训练的无监督域自适应**
  
- **基于半监督框架的无监督域自适应**
  
- **基于Soft-Teacher的无监督域自适应**

- **基于传统域自适应方法的图像转换**

  
</details>


## 实验结果
目前所有的实验结果均保存到score.csv文件中
所有的测试文件结果均在test_result文件夹中


## 更新日志
### 2023-6 至 2023-8
#### 新增
* 复现并统一完善了 DA-Faster RCNN，SCL，SWDA，HTCN，Unbiased teacher，Adaptive teacher，SADA，MIC的所有代码
* 增加了 ADAPT2023，Woodscape，SynWoodscape，ACDC数据集的相关settings
* 感谢所有人的努力和付出，并欢迎新的博士和学弟学妹加入实验室视觉祖

### 2023-6
#### 新增
* 完善了现阶段所有的数据集及配置文件
* 重构了domain adaption模块的代码
* 上传了目前所有实验的实验结果
* 算法库稳定到3.0版本

### 2023-3
#### 新增
* 增加了deformable-detr和dino的适配和配置文件
* 适配到mmdet-3.0rc6，增加了TTA的设置
* 统一了所有的DA分类器和Domain Adaption架构，目前所有的算法均在同一个文件中
* 增加了albu-domain-adaption的图像转换增强

### 2023-2-24
#### 新增
* da分类器增加了init

### 2023-2-12
#### 新增
* 增加了unity-ship to airbus-ship和synscapes to cityscapes相关设置
* 统一了数据集的名称
* 增加了Synscapes代码处理文件


### 2023-1-20
#### 新增
* 增加了sim10k->CityScapes和sim10k->BDD100k的相关配置文件
* 增加了BDD100k的数据处理的数据处理代码，与CityScapes对齐
#### 修改
* 统一了文件夹名称


### 2023-1-18
#### 新增
* 增加了ATSS，Deformalable Detr的相关配置文件
* 增加了source和target相关的训练配置文件
* 增加了RarePlanes的处理代码
#### 修改
* 统一了文件夹名称

### 2023-1-17
#### 修改
* 统一了end2end训练环境，现在不需要修改底层文件
* 删除了mmengine需要修改的文件

### 2023-1-14
#### 修改
* 统一了域分类器的实现和域分类器设置，增加了loss_type设置
* 修改了域分类器的卷积stride=1，防止出现下采样中的训练错误


### 2022-12-30
#### 新增
* 完成github代码库创建和基本代码上传
* 增加部分代码的注释和说明
