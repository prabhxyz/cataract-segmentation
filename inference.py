import os
import torch
import cv2
import numpy as np

from swin_unet import SwinUNet

FIXED_SIZE = 384

def run_inference(json_path, img_path, checkpoint="checkpoints/best_model.pth", output_path="inference_overlay.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinUNet(num_classes=3)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # 1) Read image
    frame_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        print("Cannot read image:", img_path)
        return
    orig_h, orig_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize
    frame_rgb_384 = cv2.resize(frame_rgb, (FIXED_SIZE, FIXED_SIZE), interpolation=cv2.INTER_LINEAR)

    # 2) Convert to torch
    inp = torch.from_numpy(frame_rgb_384.transpose(2,0,1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(inp)  # => [1,3,384,384]
        pred   = torch.argmax(logits, dim=1)  # => [1,384,384]

    pred_np_384 = pred.squeeze(0).cpu().numpy().astype(np.uint8)
    # Resize back to original
    pred_np = cv2.resize(pred_np_384, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # We have 0=Iris,1=Pupil,2=Lens. Let's color them distinctly.
    palette = {
        0: (0, 255, 0),    # Iris = green
        1: (255, 0, 0),    # Pupil = blue
        2: (0, 0, 255)     # Lens = red
    }
    color_mask = np.zeros_like(frame_bgr)
    for c, color in palette.items():
        color_mask[pred_np==c] = color

    overlay = cv2.addWeighted(frame_bgr, 0.6, color_mask, 0.4, 0)
    cv2.imwrite(output_path, overlay)
    print(f"Inference saved to: {output_path}")

if __name__=="__main__":
    # Example usage:
    # You pick one "case_XXXX_01.png" from your dataset. Provide the .json path too.
    case_dir = "Cataract-1k/Annotations/Images-and-Supervisely-Annotations/case_0001"
    json_path = os.path.join(case_dir, "ann", "case5013_01.png.json")
    img_path  = os.path.join(case_dir, "img", "case5013_01.png")

    run_inference(json_path, img_path)
