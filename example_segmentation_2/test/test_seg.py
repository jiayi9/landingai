# Import testing modules
import sys
import numpy as np
from landinglens.model_iteration.sdk import DataItem, checks, read_image

# Import your custom transforms
sys.path.append(".")
from custom.segmentation_to_classification import SegmentationToClassification  # noqa


def main():
    """
    Add description to test file
    """
    # You can use inputs = llens_test.create_template_dataitem()
    # or use your own logic for inputs.

    # Change to your paths
    checks.create_data()
    image_path = "test/data/mario-nes.png"
    mask_labels_path = "test/data/mario-nes_seg.png"

    # Load inputs
    image = read_image(image_path)
    label = np.random.randint(5)
    score = np.random.random()
    mask_labels = read_image(mask_labels_path)
    mask_scores = (np.arange(5) == mask_labels[..., None]).astype(int)
    defect_map = {i: f"Class_{i}" for i in range(5)}

    # Create DataItem object
    inputs = DataItem(
        image=image,
        label=label,
        score=score,
        mask_scores=mask_scores,
        mask_labels=mask_labels,
        user_data={"defect_map": defect_map},
    )

    # Instantiate the transform and call it.
    custom_transform = SegmentationToClassification()
    output = custom_transform(inputs)

    # Use here provided check functions or use your own assertions.
    checks.check_data_item(output, task="segmentation")
    print("Checks passed successfully.")


if __name__ == "__main__":
    main()
