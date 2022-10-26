from landinglens.model_iteration.sdk import BaseTransform, DataItem
import numpy as np

class NoiseFilter(BaseTransform):
      """
      """

    def __init__(self, pixel_area_threshold=None, percentage_area_threshold=None, **params):
        self.pixel_area_threshold = pixel_area_threshold
        self.percentage_area_threshold = percentage_area_threshold

        if self.pixel_area_threshold and self.percentage_area_threshold:
            raise ValueError("Just specify one threshold.")

        if not self.pixel_area_threshold and not self.percentage_area_threshold:
            raise ValueError("You have to specify one threshold.")

        if self.percentage_area_threshold and not (0 <= self.percentage_area_threshold <= 1):
            raise ValueError("Percentage is not between 0 and 1")

        if self.pixel_area_threshold and not (self.pixel_area_threshold > 0) or not isinstance(self.pixel_area_threshold, int):
             raise ValueError("Pixel value is not positive integer")

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
        img_h, img_w, _ = inputs.image.shape
        total_pixels = img_h * img_w
        bboxes = inputs.bboxes
        scores = inputs.bboxes_scores
        labels = inputs.bboxes_labels

        if bboxes is None and labels is None:
            return DataItem(**dict(inputs._asdict()))

        new_boxes = np.array([]).reshape(-1, 4)
        new_scores = np.array([])
        new_labels = np.array([])

        for box, score, label in zip(bboxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            h = y_max - y_min
            w = x_max - x_min
            pixels = h * w
            perc_area = pixels / total_pixels

            if self.pixel_area_threshold and pixels > self.pixel_area_threshold:
                new_boxes = np.vstack([new_boxes, box])
                new_scores = np.append(new_scores, score)
                new_labels = np.append(new_labels, label)
            elif self.percentage_area_threshold and perc_area > self.percentage_area_threshold:
                new_boxes = np.vstack([new_boxes, box])
                new_scores = np.append(new_scores, score)
                new_labels = np.append(new_labels, label)
            else:
                   continue

        return DataItem(
            image=inputs.image,
            bboxes=new_boxes,
            bboxes_labels=new_labels,
            bboxes_scores=new_scores,
        )
