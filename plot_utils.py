import json
import matplotlib.pyplot as plt

def save_results_to_json(results_dict, output_path="evaluation_results.json"):
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved results to {output_path}")

def plot_segmentation_metrics(results_dict, output_path="evaluation_metrics.png"):
    """
    results_dict might have:
     {
       "iou_per_class": [...],
       "mean_iou": ...,
       "dice_per_class": [...],
       "mean_dice": ...,
       "precision_per_class": [...], etc.
     }
    We'll plot IoU + Dice (and optionally more).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    classes = ["Iris", "Pupil", "Lens"]
    iou_pc = results_dict.get("iou_per_class", [])
    dice_pc= results_dict.get("dice_per_class", [])
    mean_iou= results_dict.get("mean_iou", 0)
    mean_dice=results_dict.get("mean_dice", 0)

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # IoU bar
    axes[0].bar(classes, iou_pc, color='blue')
    axes[0].set_title(f"IoU (mean: {mean_iou:.3f})")
    axes[0].set_ylim([0,1])

    # Dice bar
    axes[1].bar(classes, dice_pc, color='green')
    axes[1].set_title(f"Dice (mean: {mean_dice:.3f})")
    axes[1].set_ylim([0,1])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved metric plot to {output_path}")
