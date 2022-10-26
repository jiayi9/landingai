from landinglens.model_iteration.sdk import BaseTransform, DataItem
import numpy as np


class ObjectDetectionToClassification(BaseTransform):
    """Transforms an object detection output into a classification output by simply
    taking the bounding box with the highest score given. If the key 'label' is passed
    keyword argument, this transform will ignore it.
    """

    def __init__(self, highest_score_position: int = 1, **params):
        """Any parameters defined in yaml file will be passed to init. Store the value
        passed in self so you can access them in the call method.

        Parameters
        ----------
        highest_score_position : int = 1
            The position of the highest score bounding box to define it as
            the classification image label.
        """
        # TODO: Add here your initialization parameters (if no parameters just put ´pass´)
        assert isinstance(
            highest_score_position, int
        ), f"Score position is not an integer. Got {highest_score_position}"
        assert (
            highest_score_position > 0
        ), f"Score position is not positive. Got {highest_score_position}"
        self.highest_score_position = highest_score_position

    def __call__(self, inputs: DataItem) -> DataItem:
        """Return a new DataItem with transformed attributes. DataItem has following
        attributes:

        image - input image.
        label, score - input label and its score.
        mask_scores, mask_labels - segmentation mask probabilities and classes.
        bboxes, bboxes_labels, bboxes_scores - object detection bounding boxes.
        user_data - any additional data that you want to store for subsequent transform.

        Returns
        -------
            A named tuple class DataItem with the modified attributes.
        """
        # TODO: Add custom transform code here to modify the targets.
        # Get scores and label and raise error if not defined
        scores = inputs.bboxes_scores
        labels = inputs.bboxes_labels

        if scores is None:
            raise TypeError("'bboxes_scores' not defined in inputs")
        if labels is None:
            raise TypeError("'bboxes_labels' not defined in inputs")

        # If no scores then OK, else sort and get given position highest scored label.
        if scores.size == 0:
            label = 0
            score = 0
        else:
            # Sorting
            arg_sort = np.argsort(scores)[::-1]
            scores = scores[arg_sort]
            labels = labels[arg_sort]
            # Get label
            score = float(scores[self.highest_score_position - 1])
            label = int(labels[self.highest_score_position - 1])

        return DataItem(
            image=inputs.image,
            label=label,
            score=score,
            bboxes=inputs.bboxes,
            bboxes_labels=inputs.bboxes_labels,
            bboxes_scores=inputs.bboxes_scores,
        )
