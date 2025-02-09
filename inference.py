import argparse
import os
import cv2
import torch
import numpy as np

from models import get_segmentation_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True,
                        help="Path to an input .mp4 video.")
    parser.add_argument('--timestamp', type=float, required=True,
                        help="Timestamp (in seconds) for extracting a frame.")
    parser.add_argument('--checkpoint', type=str, default="best_model.pth",
                        help="Path to your trained model checkpoint.")
    parser.add_argument('--output_dir', type=str, default="output",
                        help="Where to save output images.")
    parser.add_argument('--encoder_name', type=str, default="resnet101",
                        help="Encoder name used in training.")
    parser.add_argument('--encoder_weights', type=str, default="imagenet",
                        help="Encoder weights used in training.")
    return parser.parse_args()

def add_legend(image, label, color, start_x=10, start_y=10, box_size=30,
               spacing=10, font_scale=0.7, thickness=2):
    legend = image.copy()
    top_left = (start_x, start_y)
    bottom_right = (start_x + box_size, start_y + box_size)
    cv2.rectangle(legend, top_left, bottom_right, color, thickness=-1)
    text_x = start_x + box_size + spacing
    text_y = start_y + box_size - 5
    cv2.putText(legend, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return legend

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load video, extract frame
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {args.video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(args.timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: cannot extract frame at {args.timestamp} sec.")
        return
    original_size = (frame.shape[1], frame.shape[0])
    original_path = os.path.join(args.output_dir, "original.png")
    cv2.imwrite(original_path, frame)
    print(f"Saved original frame to {original_path}")
    cap.release()

    # Convert BGR->RGB, resize to 512
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_size = (512, 512)
    img_resized = cv2.resize(frame_rgb, input_size)
    img_tensor = torch.from_numpy(img_resized.transpose(2,0,1)).float().unsqueeze(0)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_segmentation_model(num_classes=1,
                                   encoder_name=args.encoder_name,
                                   encoder_weights=args.encoder_weights)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        output = model(img_tensor.to(device))
        output = torch.sigmoid(output)

    # threshold => [512,512]
    mask = output.squeeze(0).squeeze(0).cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)*255
    mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Save binary mask
    mask_path = os.path.join(args.output_dir, "Cornea.png")
    cv2.imwrite(mask_path, mask_resized)
    print(f"Saved binary cornea mask to {mask_path}")

    # Overlay
    overlay = frame.copy().astype(np.float32)
    color = np.array([0,255,0], dtype=np.float32) # green
    alpha = 0.5
    bool_mask = (mask_resized == 255)
    overlay[bool_mask] = overlay[bool_mask]*(1-alpha) + color*alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_legend = add_legend(overlay, "Cornea", (0,255,0))
    overlay_path = os.path.join(args.output_dir, "overlay.png")
    cv2.imwrite(overlay_path, overlay_legend)
    print(f"Saved overlay to {overlay_path}")

if __name__ == "__main__":
    main()