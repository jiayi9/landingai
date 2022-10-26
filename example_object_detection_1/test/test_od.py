# Import testing modules
import sys
import numpy as np
from landinglens.model_iteration.sdk import DataItem, checks, read_image, read_pascalvoc

# Import your custom transforms
sys.path.append(".")
from custom.object_detection_to_classification import (  # noqa
    ObjectDetectionToClassification,  # noqa
)  # noqa


def main():
    """
    Add description to test file
    """
    # You can use inputs = llens_test.create_template_dataitem()
    # or use your own logic for inputs.

    # Change to your paths
    checks.create_data()
    image_path = "test/data/mario-nes.png"
    bboxes_path = "test/data/mario-nes_pascalvoc.xml"

    # Load inputs
    image = read_image(image_path)
    label = np.random.randint(5)
    score = np.random.random()
    defect_map = {i: f"Class_{i}" for i in range(5)}
    bboxes, bboxes_labels = read_pascalvoc(bboxes_path, output_format="pascalvoc")
    bboxes_labels = bboxes_labels.astype(np.int)
    bboxes_scores = np.random.random(bboxes_labels.shape)
    # Normalize scores to sum up 1
    bboxes_scores /= bboxes_scores.sum(axis=-1)

    # Create DataItem object
    inputs = DataItem(
        image=image,
        label=label,
        score=score,
        bboxes=bboxes,
        bboxes_labels=bboxes_labels,
        bboxes_scores=bboxes_scores,
        user_data={"defect_map": defect_map},
    )

    # Instantiate the transform and call it.
    custom_transform = ObjectDetectionToClassification()
    output = custom_transform(inputs)

    # Use here provided check functions or use your own assertions.
    checks.check_data_item(output, task="object-detection")
    print("Checks passed successfully.")


if __name__ == "__main__":
    main()
