# Custom Transforms SDK

This document explains how to create a use the LLens SDK and LandingLens CLI to create custom transformations.

You will need to have the working folder initialized. To do this, go to the [README](../README.md) file in the root
repo or in the [github repository](https://github.com/landing-ai/landinglens-cli).

## Contents

1. [DataItem Definition](#DataItem)
2. [Create Transforms](#Create-Transforms)
3. [Test Transforms](#Test-Transforms)
4. [YAML Configuration and Postprocessing](#YAML-Configuration-and-Experiment-Type)
5. [Evaluation on TRT](#Evaluation-On-TRT)

We'll first start introducing basic transformations concepts.

## DataItem

DataItem is a main concept of the transforms input since it represents any item of interest in the developer to modify
or use in one sample of data. There are 4 categories of data in transforms right now: image, label, segmentation mask and bounding boxes.

### Image

This is the dataset source image as a numpy array with 3 dimensions (height, width, Channels).
Usually the channels for the image is 3, meaning an RGB image but if your images are grayscale channels will be 1.

This can be accessed by calling `DataItem.image`.

### Label & Score

This category holds the image level label for the image, that is a Python integer indicating the class index of the defect map
belonging to that image, e.g. 0 for "OK", 1 for "Class 1", etc. Also, can include the associated confidence score of
the label as a float between 0 and 1.

**NOTE**: The integer should be in the defect map, otherwise the job will crash.

```python
label = 3 # Defect map: {0: "OK", 1: "Class 1", 2: "Class 2", 3: "Class 3"}
score = 0.6 # This can be None
```

These can be accessed by calling `DataItem.label` and `DataItem.score`.

### Segmentation Mask

This category contains the segmentation mask for the image in the segmentation task. There are two attributes of the mask:

1. **mask_scores**: This mask contains the probability per class per pixel. This is a numpy array with 3 channels
(height, width, classes).

2. **mask_labels**: This mask contains each pixel index class (argmax of the mask scores) where each index is on the defect map.
This is a numpy array with 2 channels (height, width).

An example of this category if the segmentation model outputs three possible classes, then mask are:

```python
import numpy as np
# Scores per pixel (2x2x3)
mask_scores = np.array([[[0.9 , 0.1 , 0.  ], [0.  , 0.6 , 0.4 ]],
                        [[0.03, 0.95, 0.02], [0.8 , 0.2 , 0.  ]]])
# Labels per pixel (2x2)
mask_labels = np.array([[0, 1],
                        [2, 0]])
```

These can be accessed by calling `DataItem.mask_scores` and `DataItem.mask_labels`.

### Bounding Boxes

Finally, this category contains the bounding boxes for the object detection task and it has 3 attributes: boxes, classes and scores.

1. **bboxes:** This attribute contains the bounding boxes for the object-detection task as a 4-dimensional array with the
   x-min, y-min, x-max, y-max coordinates of each box. These coordinates come relative to the bottom left corner of the
   image (PascalVOC Format).

2. **bboxes_labels:** This attribute contains the class for each of the bounding boxes of the first attribute as a 1-dimensional
   array with the integer class index in the defect map.

3. **bboxes_scores:** This attribute contains the score for each of the bounding boxes of the first attribute as a 1-dimensional
   array with the float score between 0 and 1 indicating the probability of a defect inside the box. For GT labels, it can be all
   1's or no score attribute.

An example of these three attributes of the bounding boxes is:

```python
import numpy as np
# Two bounding boxes. Image 500x500
bboxes = np.array([[100., 50., 200., 150.],
                   [175., 60., 445., 100.]])
# Class 4 for first bbox and class 1 for second bbox
bboxes_labels = np.array([[4], [1]])
# Score probability of 0.8 and 1 for bboxes respectively
bboxes_scores = np.array([[0.8], [1.0]])
```

These can be accessed by calling `DataItem.bboxes`, `DataItem.bboxes_scores`  and `DataItem.bboxes_labels`.

### Prediction vs Ground Truth DataItem

There's a little difference when the data item is fed to a prediction vs. ground truth pipeline. For the model prediction
pipeline, the DataItem comes in the format shown above. When the DataItem is for ground truth pipeline, then `_scores` keys
comes as a one-hot encoded (0's and 1's) version of the scores. This is done, so the same custom transforms can be applied
indistinctly between both pipelines. If you wish to configure different pipelines for ground truth and prediction,
please refer to this [section](#Prediction-vs-Ground-Truth-Pipelines)

```python
import numpy as np
# Ground truth scores per pixel (2x2x3)
mask_scores = np.array([[[1.0 , 0., 0. ], [0. , 1.0 , 0. ]],
                        [[0. , 1.0, 0. ], [1.0, 0.  , 0. ]]])
```

```python
import numpy as np
# Ground truth score probability of 1.0 for each bounding box.
bboxes_scores = np.array([[1.0], [1.0]])
```

## Create Transforms

Now that you got the DataItem concept, to start creating new transforms, a template file `custom/Template.py` was generated,
so you can base your developing on top of it.

### Step 1: Define the attributes

Just like a regular Python class, you can modify the name of the transform, e.g., `FilterClasses`. **Remember to always
inherit from the given *BaseTransform* class.** It's recommended to add docstring specifying what the transform does and what items are changed. Let's rename the file
as `filter_classes.py` inside the `custom` folder. This is important as we will reference the location and names
when adding the step to yaml configuration file.

In this step you can define the class attributes (with optional default values) that gets passed from the yaml file
`params` for the custom transform.

```python
from landinglens.model_iteration.sdk import BaseTransform, DataItem

class FilterClasses(BaseTransform):
    """
    This transforms filter classes from the mask, and if the specified class is present in the mask then image
    label is NG, else OK.
    """

    def __init__(self, filter_classes = [1, 3], ng_class = 2):
        self.filter_classes = filter_classes
        self.ng_class = ng_class
```

### Step 2: implement call method

You will need to write your custom logic inside the `__call__` method. For this, the argument `inputs` is passed with
all the relevant categories for the task you're working on. This is a custom named tuple object (`DataItem`) that you
can extract the relevant data items using the dot notation, i.e., if you want the image you have to call `inputs.image`
or if you want the bounding boxes coordinates you have to call `inputs.bboxes`.

You can use these inside `inputs` to write your custom logic and at the end just return also a new `DataItem`
with the important categories.

#### DataItem Specifications

- inputs.image: numpy array of shape (H, W, 3) of type np.float32.
- inputs.mask_scores: numpy array of shape (H, W, C) of type np.float32.
- inputs.mask_labels: numpy array of shape (H, W) of type np.int32.
- inputs.label: int indicating class index.
- inputs.score: float indication probability score.
- inputs.bboxes: numpy array of bounding boxes in pascal_voc format (x_min, y_min, x_max, y_max).
- inputs.bboxes_labels: numpy array of labels for each bounding box.
- inputs.bboxes_scores: numpy array of scores (0-1) for each bounding box.
- inputs.user_data: A python dictionary to store any keys other than this list that you want to access in subsequent transforms.

```python
    def __call__(self, inputs: DataItem) -> DataItem:
        """
        This method implement the filtering and the defines the label if self.ng_class is present for segmentation.
        """
        # Checks if mask_labels is defined, if not then declare it as argmax of the scores.
        if inputs.mask_labels is None:
            inputs.mask_labels = np.argmax(inputs.mask_scores, axis=-1)

        # Filtering mask classes based on self.filtering_classes.
        is_filtering = np.isin(inputs.mask_labels, self.filter_classes)
        new_mask = np.where(is_filtering, 1, inputs.mask)

        # If mask contains self.ng_class then NG else OK.
        is_ng = np.isin(inputs.mask_labels, self.ng_classs)
        final_label = 1 if any(is_ng) else 0 # defect map = {0: "ON", 1: "NG"}

        # Return new DataItem with relevant categories.
        return DataItem(
            image=inputs.image,
            label=final_label,
            mask_labels=new_mask,
        )
```

And that's it! You have written your own custom transform!

## Test Transforms

We are implementing a local testing framework. We'll keep you informed.

## Custom Transforms to YAML

To add it in a YAML file, you must follow the format:

```yaml
- CustomTransform:
  transform: folder.file_name.ClassName
  params:
    param_1: "This is Parameter 1"
    param_2: 4
```

Where `folder` can be multiple folders separated by `.`, and params are the initialization arguments from your custom transform.

For the `FilterClasses` we created, the YAML config is the following:

```yaml
- CustomTransform:
  transform: custom.filter_classes.FilterClasses
  params:
    filter_classes: [1, 3]
    ng_class: 2
```

## YAML Configuration and Experiment Type

### Preprocessing

For preprocessing just add the custom transform configuration in the `transform.yaml` file under
the `train` or/and `valid` key just as a regular transform.

```yaml
# transforms.yaml
train:
  - CustomTransform:
      transform: custom.filter_classes.FilterClasses
      params:
        filter_classes: [1, 2, 3]
        ng_class: 5
  - HorizontalFlip:
      p: 0.5
  - VerticalFlip:
      p: 0.5
  - GaussianBlur:
      p: 0.5
```

### Postprocessing and Experiment Report

For postprocessing, put this configuration in the `eval.postprocessing.transform` key of the training yaml file and
configure the experiment report settings.

#### Classification

For the classification experiment report type just add `output_type: classification` to `eval.postprocessing` key
and make sure your custom transform is returning the `label` (optionally `score`).

```yaml
# train.yaml
...
eval:
  postprocessing:
    output_type: classification
    transforms:
      - CustomTransform:
          transform: custom.filter_classes.FilterClasses
          params:
            filter_classes: [1, 2, 3]
            ng_class: 5
...
```

#### Object Detection

For the object detection experiment report type just add `output_type: object-detection` to `eval.postprocessing` key
and make sure your custom transform is returning the bounding boxes (`bboxes`, `bboxes_score` and `bboxes_labels`).

Also, there's a default IOU threshold of 0.5 that it's used for determining the True Positives (TP) of the experiment, i.e, that
if a predicted bounding box has an IOU greater than 0.5 wrt to the ground truth box and is the same class then is counted
as a TP. Else if IOU is less than 0.5, it will be counted as a False Positive even if it's the same class.
If you wish to override the threshold just add `iou_threshold: 0.8`

```yaml
# train.yaml
...
eval:
  postprocessing:
    output_type: object-detection
    iou_threshold: 0.7
    transforms:
      - CustomTransform:
          transform: custom.filter_classes.FilterClasses
          params:
            filter_classes: [1, 2, 3]
            ng_class: 5
...
```

### Prediction vs Ground Truth Pipelines

By default, the `transforms` key is applied to prediction and ground truth data, i.e, that the same set of transform
defined under that key are applied to both data.

If you wish to specify a different pipeline for each data type, there's a special key for each of those. For the ground truth
pipeline use `gt_transform` and for prediction pipeline use `pred_transform`.

Here are some examples of the possible configuration:

```yaml
# Same pipeline to ground truth and prediction
eval:
  postprocessing:
    transforms:
      - CustomTransform:
```

```yaml
# Different pipelines for ground truth and prediction
eval:
  postprocessing:
    # prediction pipeline
    pred_transforms:
      - CustomTransformPrediction:
    # ground truth pipeline
    gt_transforms:
      - CustomTransformGroundTruth:
```

```yaml
# Only define prediction pipeline (not apply to ground truth).
eval:
  postprocessing:
    # ground truth pipeline
    gt_transforms:
      - CustomTransform:
```

```yaml
# Only define ground truth pipeline (not apply to prediction).
eval:
  postprocessing:
    # ground truth pipeline
    pred_transforms:
      - CustomTransform:
```

## Evaluation On TRT

If you wish to run eval using a TRT model you should add the following part in `eval` key:

```yaml
  trt:
    trt_version: 5 # Defines trt_version to run (5 or 7)
    batch_size: 1
    fp16: true # Precision of the floating points operations
    eval_in_trt: true
```

Finally, modify training configuration and send the training job as shown [here](../README.md#Change-Training-Configuration).
