# from landinglens.model_iteration.sdk import BaseTransform, DataItem
# import numpy as np
#
#
# class SegmentationToClassification(BaseTransform):
#     """Transforms a segmentation output into a classification output. by simply taking
#     the label with more pixels in the segmentation mask.
#     """
#
#     def __init__(self, highest_pixel_count: int = 1, **params):
#         """Any parameters defined in yaml file will be passed to init. Store the value
#         passed in self so you can access them in the call method.
#
#         Parameters
#         ----------
#         highest_pixel_count : int = 1
#             The position of the highest pixel label count to define it as
#             the classification image label.
#         """
#         # TODO: Add here your initialization parameters (if no parameters just put ´pass´)
#         assert isinstance(
#             highest_pixel_count, int
#         ), f"Count position is not an integer. Got {highest_pixel_count}"
#         assert (
#             highest_pixel_count > 0
#         ), f"Count position is not positive. Got {highest_pixel_count}"
#
#         self.highest_score_position = highest_pixel_count
#
#     def __call__(self, inputs: DataItem) -> DataItem:
#         """Return a new DataItem with transformed attributes. DataItem has following
#         attributes:
#
#         image - input image.
#         label, score - input label and its score.
#         mask_scores, mask_labels - segmentation mask probabilities and classes.
#         bboxes, bboxes_labels, bboxes_scores - object detection bounding boxes.
#         user_data - any additional data that you want to store for subsequent transform.
#
#         Returns
#         -------
#             A named tuple class DataItem with the modified attributes.
#         """
#         # TODO: Add custom transform code here to modify the targets.
#         # Get scores and labels and raise error if not defined
#         mask = inputs.mask_scores
#
#         if mask is None:
#             raise TypeError("'mask_scores' not defined in inputs")
#
#         labels = np.argmax(mask, -1)
#
#         # Count how many labels pixel are in total and sort them
#         values, counts = np.unique(labels, return_counts=True)
#         arg_sort = np.argsort(counts)[::-1]
#         values_sorted = values[arg_sort]
#         counts_sorted = counts[arg_sort]
#
#         # Get label and score (proportion)
#         label = int(values_sorted[self.highest_score_position - 1])
#         score = float(
#             counts_sorted[self.highest_score_position - 1] / np.sum(counts_sorted)
#         )
#
#         return DataItem(
#             image=inputs.image,
#             label=label,
#             score=score,
#             mask_scores=inputs.mask_scores,
#             mask_labels=inputs.mask_labels,
#         )
