import argparse
import os
import cv2
import torch
import numpy as np
from models import get_segmentation_model

# Mapping: (Label, channel index) in the model output.
# We assume model output order: channel 0 = Iris, channel 1 = Pupil, channel 2 = Lens.
# We want to output in the order: Iris, Lens, Pupil.
CLASS_LABELS = [("Iris", 0), ("Lens", 2), ("Pupil", 1)]

# Define colors for overlay in BGR.
COLOR_MAP = {
    "Iris": (0, 255, 0),  # Green
    "Lens": (0, 0, 255),  # Red
    "Pupil": (255, 0, 0)  # Blue
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True,
                        help="Path to the input video (e.g., case_0001.mp4)")
    parser.add_argument('--timestamp', type=float, required=True,
                        help="Timestamp (in seconds) at which to extract the frame")
    parser.add_argument('--checkpoint', type=str, default="best_model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument('--output_dir', type=str, default="output",
                        help="Directory where the output images will be saved")
    parser.add_argument('--encoder_name', type=str, default='timm-efficientnet-b4',
                        help="Encoder name used in training")
    parser.add_argument('--encoder_weights', type=str, default='imagenet',
                        help="Encoder weights used in training")
    return parser.parse_args()


def add_legend(image, color_map, start_x=10, start_y=10, box_size=30, spacing=10, font_scale=0.7, thickness=2):
    """
    Draw a legend in the top-left corner of the image.
    color_map: dict mapping label -> color (BGR)
    Returns a new image with the legend drawn.
    """
    legend = image.copy()
    y = start_y
    for label, color in color_map.items():
        # Draw a filled rectangle.
        top_left = (start_x, y)
        bottom_right = (start_x + box_size, y + box_size)
        cv2.rectangle(legend, top_left, bottom_right, color, thickness=-1)
        # Put label text to the right of the rectangle.
        text_x = start_x + box_size + spacing
        text_y = y + box_size - 5  # adjust text vertical alignment
        cv2.putText(legend, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += box_size + spacing
    return legend


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Open video and extract frame at the given timestamp.
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: could not open video {args.video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = int(args.timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: could not read frame at {args.timestamp} seconds.")
        return
    original_size = (frame.shape[1], frame.shape[0])  # (width, height)
    # Save the original frame.
    original_path = os.path.join(args.output_dir, "original.png")
    cv2.imwrite(original_path, frame)
    print(f"Saved original image at {original_path}")
    cap.release()

    # Convert frame from BGR to RGB for model input.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame to model input size (assumed to be 512x512).
    input_size = (512, 512)
    img_resized = cv2.resize(frame_rgb, input_size)

    # Prepare input tensor.
    # **IMPORTANT:** Do not divide by 255 if the model was trained on images in [0, 255].
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float().unsqueeze(0)

    # Load the segmentation model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_segmentation_model(num_classes=3,
                                   encoder_name=args.encoder_name,
                                   encoder_weights=args.encoder_weights)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(img_tensor.to(device))  # shape: [1, 3, 512, 512]
        output = torch.sigmoid(output)
    output = output.squeeze(0)  # shape: [3, 512, 512]
    output_np = output.cpu().numpy()

    # Create binary masks: threshold at 0.5.
    binary_masks = (output_np > 0.5).astype(np.uint8) * 255  # values 0 or 255

    # For each label, resize the binary mask to the original size and save it.
    for label, channel in CLASS_LABELS:
        mask = binary_masks[channel]  # shape: [512,512]
        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        mask_path = os.path.join(args.output_dir, f"{label}.png")
        cv2.imwrite(mask_path, mask_resized)
        print(f"Saved binary mask for {label} at {mask_path}")

    # Create a colored overlay image.
    # Start with the original frame (in BGR).
    overlay = frame.copy().astype(np.float32)
    alpha = 0.5  # blending factor
    # For each class, overlay its color on the regions where the mask is set.
    for label, channel in CLASS_LABELS:
        # Resize mask to original size.
        mask = cv2.resize(binary_masks[channel], original_size, interpolation=cv2.INTER_NEAREST)
        # Create a boolean mask.
        mask_bool = mask == 255
        # Get the color for this label.
        color = np.array(COLOR_MAP[label], dtype=np.float32)
        # Blend: For pixels where mask is true, blend original and color.
        overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + color * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Add a legend (key) to the overlay image.
    overlay_with_key = add_legend(overlay, COLOR_MAP, start_x=10, start_y=10, box_size=30, spacing=10)

    # Save the overlay image.
    overlay_path = os.path.join(args.output_dir, "overlay.png")
    cv2.imwrite(overlay_path, overlay_with_key)
    print(f"Saved overlay image with key at {overlay_path}")


if __name__ == "__main__":
    main()