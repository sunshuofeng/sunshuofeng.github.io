---
layout: post
title: detectron2-train篇
subtitle: "学习学习！"
author: "Sun"
date: 2020-3-24
header-img:  "img/detectron2.png"
tags: 
       - pytorch
       - detectron2

---

### Hooks

detectron2是个高扩展性，灵活的框架，那么为了能够实现训练过程的灵活性，detectron2采用了hook的机制来定义训练的各个部分：
- 迭代数据集之前
- 每一次迭代batch之前
- 迭代batch进行训练
- 迭代完一个batch之后
- 结束数据集的迭代

**那么有些人可能会疑惑为啥只有迭代一次数据集呢？**

这个也是detectron2的一个特点，我们是在定义配置文件的时候进行定义epoch的数量，但epoch并不是显示去定义的，而是通过max_iter的数量，不断重复迭代数据集，比如batch=8，一共有800张图片，则迭代一次数据集需要100次迭代，若Max_iter=1000，则代表epoch=10。

*我们的重中之重就是做好run-step,找到最好的optimizer，最好的loss，最好的lr_scheduler，就行了！*

理解这一点之后我们回到Hooks,上面说到了训练的各个部分，接下来看一下Hooks的基类代码：

```python
class HookBase:
   

    def before_train(self):
        """
       迭代数据集前的步骤
        """
        pass

    def after_train(self):
        """
        迭代完数据集后的步骤
        """
        pass

    def before_step(self):
        """
        每一步迭代之前
        """
        pass

    def after_step(self):
        """
        迭代完一步之后
        """
        pass

```
对于这四个步骤要干什么就是个人设计的流程问题了，这里不多说。

**现在我们的重点就是这几个如何连接在一起呢？**

因为这四个都是独立的函数，而每一种hook的每一个功能他需要的参数可能都不一样，有些需要loss，有些需要metric，有些需要训练的图片记录下来，有些需要当前的迭代次数，有些需要当前的学习率，那么这些信息是如何进行传递呢？

我们看这段代码：
```python
  def train(self, start_iter: int, max_iter: int):
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            finally:
                self.after_train()

```
这段代码使用了一个叫做EventStorage的类，那么这个类就是传递信息的关键.


#### EventStorage
这个eventstorage就是储存信息以及传递信息的关键。

那么怎么使用呢？**首先这个EventStorage在训练过程中相当于一个全局的字典，所以我们可以在任何地方获取以及存储任何值**

要获得这么一个字典，只需调用下面的函数
```python
storage=get_event_storage()
```
那么我们就获取到了这个对象，然后就是存储以及获取信息了

**对于信息，我们分成两类：标量信息和非标量信息**

为什么要这样分呢，我们来看一下

**对于标量信息：**

我们使用put_scalar,put_scalars两个函数进行信息的储存，两者的区别就是一个一次储存一个标量，后者可以储存多个
```python
#设置标量的名字以及其值
storage.put_scalar('loss',20)

#通过关键字参数进行多个标量的存储
#不过关键字仍然是作为字符串变成字典的值
storage.put_scalars(loss=20,metric=10)


```

我们可以看一下下面这个例子：
```python
def run_step():
  storage=get_event_storage()
  print(storage.latest())
  

with EventStorage(0) as storage:
  for i in range(10):

    storage.put_scalars(loss=i*10,metric=i*5)
    storage.put_scalar('lr',i)
    run_step()
  
```
```python
{'loss': 0.0, 'metric': 0.0, 'lr': 0.0}
{'loss': 10.0, 'metric': 5.0, 'lr': 1.0}
{'loss': 20.0, 'metric': 10.0, 'lr': 2.0}
{'loss': 30.0, 'metric': 15.0, 'lr': 3.0}
{'loss': 40.0, 'metric': 20.0, 'lr': 4.0}
{'loss': 50.0, 'metric': 25.0, 'lr': 5.0}
{'loss': 60.0, 'metric': 30.0, 'lr': 6.0}
{'loss': 70.0, 'metric': 35.0, 'lr': 7.0}
{'loss': 80.0, 'metric': 40.0, 'lr': 8.0}
{'loss': 90.0, 'metric': 45.0, 'lr': 9.0}
```

那么如何获取标量信息呢？
- storage.latest()  ----->这个返回的是当前迭代时赋予的最新值（因为一直在更新这个字典）
- sotrage.histories() ---->这个返回的是HistoryBuffer对象，储存了以前所有赋值的信息

对于latest，不多说，从上面的输出就可以看出，就是返回当前字典的最新值

**对于HistoryBuffer，这个就是为什么标量信息可以单独拿出来谈论的重点**

historybuffer的处理在detectron2中并没有说到，而这个处理是在fvore(facebook的另外一个第三方库）

[HistoryBuffe文档](https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/history_buffer.py)

对于HistoryBuffer，具体就是三种操作：
```python
history=storage.histories()

#获取最新的信息
history.latest()

#获取所有标量值，返回元组（标量值，第几个迭代设置的这个标量值）
history.values()

#获取标量值的平均值

#前window_size个元素的平均值
history.avg(window_size)

#全部元素的平均值
history.global_avg()


```
就是因为提供了这些直接对标量值的操作，所以对于标量，我们就采用上面的方法进行存储读取


**对于非标量值**
对于一些非标量值，比如布尔向量，存储某些重要信息的字典甚至是图片，我们都会存储到EventStorage中

图片比较特殊，EventStorage本身就提供了存储图片读取图片的方法
```python
#存储图片
storage.put_image(image_name,image)

#读取图片
#返回的是[(image_name,image,iteration),....]
storage.vis_data()

```
通常存储图片是为了可视化，没什么情况不需要使用

而对于其他一些重要的非标量值
```python
storage.early_stop=True

storage.inforamtion={}
```

读取也是直接引用对应的变量名即可

```python
if storage.early_stop:
      break
```

这就是各个hook之间，hook的函数之间，进行数据传输的过程。



### 训练模型两件套：optimizer,lr_scheduler

没错，到了重要的两件套时刻。对于一个模型，最终泛化性的好坏，这两件套的重要性不言而喻。接下来我们就探讨一下detectron2这两件套是怎么样去使用的。（其实真正是三件套，还缺一个loss，但是loss是跟model挂钩的，而model的部分非常难，还会单独写一个博客来介绍model和loss，所以这里先不介绍loss了)

**optimizer**

在介绍如何定义自己的optimizer之前，同样的按照老方法我们先看一看detectron2官方默认的：
```python

def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if isinstance(module, norm_module_types):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            elif key == "bias":
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer
```
我们可以看一下官方的定义：

首先它是循环model的各个部分，对于不需要梯度下降的部分不予理会，然后根据config文件确定该部分的learning_rate,weight_decay。对于Bias部分，还做了特殊的处理，同样也是根据config进行处理的。

然后就建立了一个SGD,然后做了一个给optimizer加上梯度裁剪功能的操作，这样optimizer在进行更新参数时还会对梯度进行裁剪。

**综上所述，对于实现一个optimizer，咱们要做的不多，第一就是在config文件确定好学习率，weight_decay以及其他重要参数。第二就是确定使用哪个optimizer**

上面这些确定后，其他都可以保持不变，这样我们就建立了一个自定义的optimizer


**lr_scheduler**
lr_scheduler比起更加需要自定义的部分，我们先来看一下官方的代码：
```python

def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
  
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

```
可以看出官方提供两种的lr_scheduler,一个是余弦退火，一个是根据step进行退火，接下来我们看看提供的这两种scheduler
```python

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()

```
可以看出scheduler是通过get_lr返回某个时刻更新后的学习率。

把目光聚焦到余弦退火的scheduler,可以看出跟torch.optimi.lr_scheduler基本差不多，只不过它要一些参数要通过config实现。而最重要的更新学习率同样有get_lr函数实现

**所以自定义的scheduler就是更改config文件的配置以及重写自己的get_lr方法**

而如何通过scheduler进行学习率的更新呢？由于学习率的更新需要迭代次数的信息，所以学习率更新通常放在hook中，在run_step后通常是after_step进行
```python
lr_scheduler.step()
```
这一步。

我们也可以看到，detectron2官方本来就提供了lr_scheduler的hook，只要把optimizer和lr_scheduler传进去，就可以进行学习率的更新了
```python

class LRScheduler(HookBase):
    """
    A hook which executes a torch builtin LR scheduler and summarizes the LR.
    It is executed after every iteration.
    """

    def __init__(self, optimizer, scheduler):
        """
        Args:
            optimizer (torch.optim.Optimizer):
            scheduler (torch.optim._LRScheduler)
        """
        self._optimizer = optimizer
        self._scheduler = scheduler

        # NOTE: some heuristics on what LR to summarize
        # summarize the param group with most parameters
        largest_group = max(len(g["params"]) for g in optimizer.param_groups)

        if largest_group == 1:
            # If all groups have one parameter,
            # then find the most common initial LR, and use it for summary
            lr_count = Counter([g["lr"] for g in optimizer.param_groups])
            lr = lr_count.most_common()[0][0]
            for i, g in enumerate(optimizer.param_groups):
                if g["lr"] == lr:
                    self._best_param_group_id = i
                    break
        else:
            for i, g in enumerate(optimizer.param_groups):
                if len(g["params"]) == largest_group:
                    self._best_param_group_id = i
                    break

    def after_step(self):
        lr = self._optimizer.param_groups[self._best_param_group_id]["lr"]
        self.trainer.storage.put_scalar("lr", lr, smoothing_hint=False)
        self._scheduler.step()

```

未完待续........
