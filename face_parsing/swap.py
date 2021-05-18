import torch
import torchvision.transforms as transforms
import cv2

from model import BiSeNet


def init_parser(pth_path):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(pth_path))
    net.eval()
    return net


def image_to_parsing(img, net):
    img = cv2.resize(img, (512, 512))
    img = img[:,:,::-1]
    img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
    img = torch.unsqueeze(img, 0)

    with torch.no_grad():
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing


def get_mask(parsing, classes):
    res = parsing == classes[0]
    for val in classes[1:]:
        res += parsing == val
    return res


def swap_regions(source, target, net):
    parsing = image_to_parsing(source, to_tensor, net)
    face_classes = [1, 11, 12, 13]

    mask = get_mask(parsing, face_classes)
    mask = np.repeat(np.expand_dims(mask, axis=2), 3, 2)
    result = (1 - mask) * source + mask * target
    return result
