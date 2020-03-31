---
title: detectron2-配置篇
subtitle: "即将学完"
author: "Sun"
date: 2020-4-1
header-img: "img/detectron2.png"
tags: 
       - pytorch
       - detectron2

---

前几篇文章基本已经将整个项目的流程梳理了一遍，但是其实都缺了一个非常重要的步骤就是配置，在这篇博客中，咱们梳理一下config文件中一些重要的部分，以及各个流程必须要设置的一些部分。


**获取默认的配置**
```python
from detectron2.config import get_cfg
cfg=get_cfg()

```

### data配置
对于data，我们通常配置的就是训练集和测试集的名字，以及dataloader的一些参数

```python
#配置训练集，用于在train_dataloader里
cfg.DATASETS.TRAIN = ("balloon_train",)

#配置测试集，用于在defaultpredictor里用
cfg.DATASETS.TEST = ("balloon_val", )

#配置num_workers
cfg.DATALOADER.NUM_WORKERS = 2

#配置batch
cfg.SOLVER.IMS_PER_BATCH

```

### 训练的配置
```python
#配置模型权重，但是得保证配置文件里的默认模型跟其相同
cfg.MODEL.WEIIGHTS=model_zoo.get_checkpoint()

#所以我们通常会这样做来引入预训练模型
cfg.merge_from_file(model_zoo.get_config_file(model.pt))
cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint(mode.pt)

#配置num_class
cfg.MODEL.ROI_HEADS.NUM_CLASSES= num_class

#配置roi的minibatch，不是必须，调整这个会改变训练速度
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=512 (默认值)

#配置基本的学习率，建议哪怕是自定义optimizer,也用这个来获取学习率
cfg.SOLVER.BASE_LR=0.00025

#配置动量
cfg.SOLVER.MOMENTUM=0.9

#配置权重衰减
cfg.SOLVER.WEIGHT_DECAY=0.0001


#由于默认为warmup_scheduler,所以要根据max_iter来配置好warmup的迭代书
cfg.SOLVER.WARMUP_ITERS=100

#配置迭代数据次数，epoch也是由这个来定的,epoch=max_iter/iter一次数据集
cfg.SOLVER.MAX_ITER=300

#配置checkpoint的位置
cfg.OUTPUT_DIR

#训练的图片的尺寸上下限
cfg.INPUT.MIN_SIZE_TRAIN=(800,)
cfg.INPUT.MAX_SIZE_TRAIN=1300



```

**Test，evaluate**
```python
#将训练的模型拿过来
cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,'model_final.pth')

#设置iou的阈值
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7


```

在训练过程中基本就是这些需要设置的了。

未完待续

