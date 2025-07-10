import torch
import cv2
import numpy as np


midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()


def midas_depth(frame, scratcher=3000):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_np = np.array(img_rgb)
    result = transform(img_np)
    input_tensor = result["image"].to(device) if isinstance(result, dict) else result.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)
        elif prediction.dim() == 2:
            prediction = prediction.unsqueeze(0).unsqueeze(0)

        prediction_resized = torch.nn.functional.interpolate(
            prediction,
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False
        )

    depth_map = prediction_resized.squeeze().cpu().numpy()
    inverted_depth = scratcher * (1 / (depth_map + 1e-6))
    return inverted_depth


def depth_to_heatmap(depth_map):
    d_min, d_max = np.min(depth_map), np.max(depth_map)
    norm_depth = (depth_map - d_min) / (d_max - d_min + 1e-8)
    heatmap = (255 * (1 - norm_depth)).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)
