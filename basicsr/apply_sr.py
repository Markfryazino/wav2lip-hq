import cv2
import numpy as np
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet


def init_sr_model(model_path):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.cuda()
    return model


def enhance(model, image):
    img = image.astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(img)
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output
