import sys
import glob
import torchvision.transforms.functional as F
sys.modules['torchvision.transforms.functional_tensor'] = F

from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import cv2
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Find latest version
versions = sorted(glob.glob('vg_v*.jpg'))
if not versions:
    src = 'vg.jpg'
    next_v = 1
else:
    src = versions[-1]
    next_v = int(src.split('_v')[1].split('.')[0]) + 1

dst = f'vg_v{next_v}.jpg'
print(f'{src} -> {dst}')

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
state_dict = torch.hub.load_state_dict_from_url(
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    map_location='cpu',
)
if 'params_ema' in state_dict:
    state_dict = state_dict['params_ema']
model.load_state_dict(state_dict, strict=True)
model.eval().to(device)

img = cv2.imread(src, cv2.IMREAD_COLOR)
h, w = img.shape[:2]

# Tile-based inference to avoid OOM
scale = 4
tile_size = 256
pad = 10
out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)

for y in range(0, h, tile_size):
    for x in range(0, w, tile_size):
        # Input tile with padding
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + tile_size + pad, w)
        y1 = min(y + tile_size + pad, h)

        tile = img[y0:y1, x0:x1]
        tile_t = torch.from_numpy(tile.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            result = model(tile_t)

        result = result.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)

        # Crop padding from output
        op = pad * scale
        oy0 = 0 if y0 == 0 else op
        ox0 = 0 if x0 == 0 else op
        oy1 = result.shape[0] if y + tile_size >= h else result.shape[0] - op
        ox1 = result.shape[1] if x + tile_size >= w else result.shape[1] - op

        # Place in output
        dy0 = y * scale
        dx0 = x * scale
        dy1 = dy0 + (oy1 - oy0)
        dx1 = dx0 + (ox1 - ox0)
        out[dy0:dy1, dx0:dx1] = result[oy0:oy1, ox0:ox1]

    print(f'  row {y}/{h}')

out = (out * 255).astype(np.uint8)
cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f'Done. Saved {dst} ({out.shape[1]}x{out.shape[0]})')
