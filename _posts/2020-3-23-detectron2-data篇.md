---
layout:  post
title:  "detectron2-data篇"
subtitle: "学习学习！"
date: 2020-3-23
author:  "Sun"
header-img: "img/detectron2.png"
tags:  
       - pytorch
       - detectron2

---
**这篇博客针对于detectron2处理数据，读取数据的部分**


### DataList
datalist是读取数据的第一步，从datalist中可以读取每一张图片的大小，路径，注释等等。

**datalist格式 -list(dict)**：
- filename: 文件完整路径
- height,width: 图片大小
- sem_seg_file_name: mask图片的完整路径
- sem_seg: 像素值对应的类别
- image_id: 每一张图片的id
- annotations：list(dict)，图片对应的注释

具体annota的要求可以看文档：[detectron2-data](https://detectron2.readthedocs.io/tutorials/datasets.html)

那么我们首先会创造一个函数，这个函数是针对对应数据集获取数据集。根据数据集的特性来制定如何从该数据集来获取所有图片的datalist。

我们看一下detectron2官方提供的voc的读取方式：

```python
def load_voc_instances(dirname: str, split: str):
  
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
          
         
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
       
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

```
可以看到datalist每一个元素就是一张图片的信息字典


### 注册datalist
当我们建立了从数据集获取datalist的函数之后，我们就要注册datalist了，在后面我们就用这个注册的datalist来建立训练测试用的数据集了。

那么为什么要进行注册呢，很简单。在一个项目中我们可能要用到多个数据集，假设只有一个那我们也有train dataset和 val dateset,test dataset之类的。

为了方便，detectron2采用注册机制，对于一个数据集，我们定义它的名字和获取datalist的方法，然后进行注册。这样detectron2就知道有名字为xxxx的datalist，是通过xxx方法从xxxx数据集中获取的

然后到后面我们就可以直接根据名字来获取其对应的datalist了

```python
from detectron2.data import DatasetCatalog

#注册voc train
train_name='voc_train'
DatasetCatalog.register(train_name,load_voc_instances(dir_name,'train')

#注册voc valid
valid_name='voc_valid'
DatasetCatalog.register(valid_name,load_voc_instances(dir_name,'valid')

#我们可以直接根据名字获取对应的dataset了
train_dicts=DatasetCatalog.get('voc_train')
valid_dicts=DatasetCatalog.get('voc_valid')

```


### MetadataCatalog
我们有时会看到这样的代码：

```python
for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
```

对于这个类，在训练过程并没有什么用，通常只是为了可视化效果。

因为我们看到datalist中只包含了class的id，并没有包含每个id对应哪个类，而可视化通常需要这些信息，所有我们通过给datalist增加元数据，相当于额外的描述来进行可视化。

这里不多介绍了，因为不是重点部分，感兴趣可以看文档（其实文档也没说啥。。。）


### Transform
现在进入了重点部分了，那么现在我们拥有了datalist，如何变成pytorch对应的dataset呢？而在dataset必不可缺的就是对数据进行数据增强，那么detectron2如何对数据进行增强呢？

我们先看一个简洁的dataset的代码

```python
class MyDataset():
    def __init__(self, dataset , map_func):
      
        self.map_func = map_func

        self._dataset = dataset
    def __getitem__(self,idx):
        idx_data = self._data[idx]
        data:dict = self.map_func(idx_data)
        return data 


```

**该代码片段来自一个detectron2的教程，这个教程写的非常好，对我学detectron2帮助非常大**：
[教程链接](https://zhuanlan.zhihu.com/c_1167741072251305984)

回到正题，那么detectron2中会通过一个map_func方法来对数据进行处理(不仅仅限于数据增强)

所以整体的流程就是获得一个datalist后，我们通过map_func对datalist的数据进行处理，然后返回一个全新的datalist,当然这个新的datalist已经跟前面的不同了，由于已经做过数据的处理，**新的datalist将会是包含处理过的图片（不是path而是图片），注释等将要用在训练测试中的数据**

现在的重点就是map_func是啥？

同样引入教程里的代码片段
```python

class MapFunc():
    def __init__(self, cfg):
        self.transform_gens : List[TransformGen] = build_transform(cfg)
    def __call__(self,data_dict):

        #读取图片啦
        image = read_image(data_dict['image_id'])
        annos = data_dict['annotation']
		
		#对数据做处理啦
        image ,annos = self.apply_transform(image , annos)
		
		#返回一个新的datalist啦
        data_dict['annotations'] = annos
        data_dict['image'] = torch.as_tensor(image)

        return data_dict
		
    def apply_transform(self, image , annos):
        tfms = []
        for g in self.transform_gens:
            tfm = g.get_transform(image)
            image = tfm.apply_image(img)
            tfms.append(tfm)
        tfms_anno = TransformList(tfms)
        
        annos = [tfms_anno(a) for a in annos]  
        return image , annos
```
通过代码实例看出来，在map_func里我们读取了图片以及其标签，然后进行数据处理，返回新的datalist。

读取数据的部分不用说，很简单，重中之重是如何数据处理呢？

**重点围绕几点**
```python
self.transform_gens=build_transform(cfg)
```
transform_gen是啥呢

很简单，我们通常会根据配置文件来进行数据增强方法的确定：
- 使用什么数据增强方法
- 方法的概率是多少
- 方法的参数是多少

根据配置文件我们就获得了一系列的数据增强方法，把他放进一个列表里(类似pytorch albumentations的compose方法）,然后后面逐个迭代进行增强。

这看起来十分有道理，但是再看代码发现，哎？为啥会有这么一段呢？
```python
for g in self.transform_gens:
      tfm=g.get_transform(image)
```
这是为啥呢？

因为有些数据增强的方法它是需要image的尺寸来进行的，就比如大火的cutmix，要根据图片大小来确定裁剪部分的大小，但是我们很少会在config文件里定义图片的大小，因为在有些任务中，输入图片的尺寸不定，我们不可能在config文件中直接定义尺寸，所以就要用get_transform吧image传进去从而获取完整的transform

我们看一下detectron2官方实现的一个transform_gen:
```python
class RandomCrop(TransformGen):


    def __init__(self, crop_type: str, crop_size):
 
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute"]
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        croph, cropw = self.get_crop_size((h, w))
        assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
        h0 = np.random.randint(h - croph + 1)
        w0 = np.random.randint(w - cropw + 1)
        return CropTransform(w0, h0, cropw, croph)
```

首先在一开始根据config文件确定crop的方法，然后根据get_transform传进来的image信息获取crop的大小，那么就可以获得一个针对该image进行crop的数据增强CropTransform方法


**我们在自定义transform_gen的时候不要忘了继承TransformGen类，然后init方法是根据config文件进行设定参数和概率，get_transform根据尺寸信息返回针对该图像的tranfrom方法**

那么了解完transform_gen后，我们再来看看transform方法
```python
class MyTransform(Transform):
    def apply_image(self, img, interp=None):
       ...
       return trans_image
    def apply_coords(self, coords):
       ...
       return trans_coords
    def apply_segmentation(self, segmentation):
 
       return trans_seg

```
对于transform方法，我们要实现三种方法，apply_image,apply_coords,apply_segemtation，用于处理图像本身，图像边界框，图像mask。


**instance**
做完transform以及一系列读取数据的操作后，我们已经获得了新的datalist，但是是不是就结束了呢？

并不是！

因为detectron2在读取数据时不是读取data_dict['annotation']，而是读取data_dict['instance']。如果我们不做mapper，那么调用默认构造dataset时，它会自动帮你去掉annotation,增加instance。但是如果我们构造了mapper，我们就需要自己去完成这一步

如何完成呢，我们可以参照默认的操作：

```python
  instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
   dataset_dict["instances"] = utils.filter_empty_instances(instances)

```
在做完transform返回新的dataset['annotation']后就可以用Instance替换annotation了

不过我们通常还会把annotaion去掉：
```python
annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
```

如果前面已经做了transform，那么我们就直接这样
```python
annos=dataset_dict.pop('annotations')

```

**总结一下：**

对于自定义数据增强，我们首先定义好一系列的transform方法用于处理图像。

然后再定义一系列的transfrom_gen方法，用于根据config和图像尺寸来获取对应的transform方法。

然后定义map_func,在函数里面读取数据，根据transform_gen获取方法transform后对数据进行处理。

最后返回一个全新的datalist，包含处理过后的image以及其注释信息


### Dataset
上面的实例代码中已经做了个简单的map_dataset,从那我们可以得知，我们对于传入的datalist进行map_func处理数据后，就会返回新的datalist。

我们现在来看看官方的map_dataset :

```python
class MapDataset(data.Dataset):
  

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )

```
这就是官方的map_dataset






<font color=red>**在最后，总结一下定义数据集的整个流程8：**</font>
- def  Get_Datalist():  ----- **定义从数据集获取datalist的函数**
- DatasetCatalog.register   ----   **注册datalist**
- dict=DatasetCatalog.get()   ---- **注册完后获取datalist**
- class MyTransform(Transform) ----     **自定义transform方法**
- class MyTransformGen(TransformGen)  ----**自定义transformgen方法**
- class My_Map_Func(MyTransfromGen):   ----**自定义map_func方法**
- ds=MapDataset(dict,My_Map_Func) ----**获取数据集**




### Sampler
Sampler是pytorch读取数据的方法，定义如何获取数据。

我们先看看pytorch的Sampler基类：
```python
class Sampler(object):
  
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
```
init方法是进行初始化，获取一些参数

iter方法就是获取数据的主要方法，返回的是索引列表，然后根据索引列表在dataset获取数据

len就是索引列表的长度

接下来我们看看detectron2中的sampler方法

- TrainingSampler  ----最简单也是最常用的sampler方法跟我们平时没啥两样
- RepeatFactorTaingingSamper  ----针对于类别极其不平衡的数据集，会将某些图片重复抽出，达到类别平衡的效果
- InferenceSampler ----故名思意，就是用于test的sampler


对于其代码我们不探究，因为sampler并没有自定义的需要，除非你的数据集很特别，不然并不需要进行sampler的自定义

而在建立dataloader的过程中，选用什么sampler完全是根据config文件夹来进行确定，除非遇上类别不平衡的数据集，否则完全可以不用管他


### Dataloader
终于到最后一步了，不过非常开心的是，最后一步也是十分简单，我们只需要把上面的部分综合起来，然后调用pytorch的dataloader的api即可啦

先看一下简洁版的整个建造dataloader的过程：

```python
def build_detection_train_loader(cfg):
    dataset_names = cfg.DATASETS.TRAIN # （'coco2017','coco2014'）
   
    dataset_dicts = [DatasetCatalog.get(dataset_name) \
                           for dataset_name in dataset_names]
    
	#这里之所以做着一步，是因为用到多个数据集时会像上面那样，所以datalist变成了list(list(dict)),这里变回list(dict)，如果只有一个，不需要这样
    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))
  
    dataset = DatasetFromList(dataset_dicts, copy=False)
   
  
    mapper = DatasetMapper(cfg, True)
   
  
    dataset = MapDataset(dataset, mapper)
 
    batch_sampler = ...

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
```

很简洁明了，我们最后看一下官方实现的build_dataloader:
```python
def build_detection_train_loader(cfg, mapper=None):
   
    num_workers = get_world_size()
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    assert (
        images_per_batch % num_workers == 0
    ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    assert (
        images_per_batch >= num_workers
    ), "SOLVER.IMS_PER_BATCH ({}) must be larger than the number of workers ({}).".format(
        images_per_batch, num_workers
    )
    images_per_worker = images_per_batch // num_workers

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        sampler = samplers.RepeatFactorTrainingSampler(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        data_loader = AspectRatioGroupedDataset(data_loader, images_per_worker)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_worker, drop_last=True
        )
        # drop_last so the batch always have the same size
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )

    return data_loader
```

**所以基本上我们在自定义数据集的情况下，需要我们自己写的应该是:**
- 获取datalist的函数
- map_func
- 改变config文件的配置

这样我们基本可以进行自定义数据集了！

Finshi!




**感谢各位能够看到最后，若有错误或者改进意见，请联系博主哟！**

另外觉得博客写的不错的同学帮忙给github加个星星哟！

