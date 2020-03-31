---
layout: post
title: "detectron2-测试评估篇"
subtitle: "马上学习完了"
author: "Sun"
date: 2020-3-30
header-img: "img/detectron2.png"
tags: 
       - pytorch
       - detectron2

---
我的前三篇博客讲述的分别是data数据的准备，训练的一些配置，以及模型的讲解，那么这篇博客将会讲解我们进行模型训练后如何进行一个评估或者是测试

咱们先从评估说起：

**Evalue**


### dataloader
首先我们需要建立jian起用于评估地dataloader，那么正如data篇里说到，建立dataloder地函数我们不需要重写，可以直接引用，**对于evalue而言，并不需要训练数据地数据增强，所以我们可以直接引用建立test_dataloader的函数，对于test_dataloader,detectron2没有做数据增强，而是做了tta，更加适合evalue**

```python
val_loader = build_detection_test_loader(cfg, "balloon_val")
```
要注意哦，在建立train_dataloader之前，我们会将数据集的名字放进config中，所以train_dataloader不需要传入注册的数据集名字，而build_test_loader是需要的。

### evaluator
我们需要建立一个（或者多个，对于多任务的训练需要多个evaluator是很正常的）evaluator来进行评估

如果只有一个evaluator,那么我们并不需要做任何处理

```python
evaluator = COCOEvaluator("balloon_val", cfg, False, output_dir="./output/")
```

但如果有多个evaluator,那么我们需要将其放入一个列表进行处理，我们可以看一下官方提供的一个代码函数：
```python
ef get_evaluator(cfg, dataset_name, output_folder=None):

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

```
可以看出，对于多个evaluator，将其放入evaluator_list中，然后返回一个DatasetEvalutor.


接下来我们就来具体看看这个evaluator是什么

看几个例子：
**coco evaluator:**

```python
class COCOEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.
    """
   
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
		
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
                " Trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

```
可以看到init方法就是定义了进行evaluate的数据集，以及配置文件，输出的一些路径等等，没有什么重要信息。

**reset方法就是将所有的预测信息清除掉,这个方法应该是一般evaluator要有的，我们可以看一下DatasetEvaluator的代码，也有这个方法**
```python
   def reset(self):
        self._predictions = []
```

这个函数应该就是根据config文件来确定评估什么任务
```python
  def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

```

**这个process函数也应该是要有的，其他evaluator也有。那么这个方法就是获取模型输入的image_id,模型输出的instance,或者proposal**
```python
 def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)
```

而这里用到一个instance_to_coco_json的函数，其实就是将输出和image_id，整合成coco的文件格式：
这里的image_id,boxes,scores是output和input传入的，而classes是数据集的元数据提供的
```python
 for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
      
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
```

**这里就是最重要的evaluate函数
```python
 def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

```
在这里面并没有什么评估的核心代码，都是一些保存文件，记录，引用函数的代码，重点就是这个eval_box_proposals和_eval_predictions

这两个代码我们并不需要多了解，我们需要了解的是这个evaluate函数输出了什么。
```python
 metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]
```
这个很形象的说明了evaluate的输出，每个任务都有对应的metric

```python
evalue_output['box']['AP50']
```

同理，其他类型的evaluate也是这样的输出。

那么不同的evaluate其实就是评估的api不同，其任务不外乎就那几种,box,seg,keypoints,关键就是你想用哪个api，当然，对于coco和pascal两个evaluate需要注意其box的模式，因为box_mode也就coco和pascal两种，其evaluate都是用对应的box_mode的。

**当然对于sem_seg的任务，我们要单独用sem_seg的evaluate**


### inference
detectron2提供了一个共用的推断函数

需要提供模型，dataloader，以及evaluator，分别就是对应上面三点了，
```python
evaluator = COCOEvaluator("balloon_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "balloon_val")
inference_on_dataset(trainer.model, val_loader, evaluator)

```


```python

def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

```
在这个函数里，通过dataloader读取数据，然后模型进行评估，获取模型的输出，然后放进evaluator,
最后进行一个evaluate，获取最后的metric字典。

这就是整个评估的过程了，非常简单，我们基本不需要重写什么，基本套用detectron2的函数就可。


**Test**
对于test，我们基本没有很多特殊的手段，基本每个项目的test都差不多，就是获取模型最后的输出结果而已，最多也就是做个tta而已，而detectron2在build_test_dataloader就做了tta，所以我们完全可以套用detectron2的predictor.

不过这里的im必须是一张图片，不过test_dataloader本来就是batch=1的dataloader，所以我们用函数建立test_dataloader，直接迭代即可
```python

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

```
然后outputs将会是一个instnce,直接获取其box坐标即可。




test和evaluate的讲解就到此了，非常简单！

觉得博客还ok的同学可以到github点个星星哟！万分感谢！



