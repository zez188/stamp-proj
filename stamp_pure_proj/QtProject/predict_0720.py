import os
import json

import torch
from PIL import Image
from torchvision import transforms
# import matplotlib.pyplot as plt

from QtProject.model import efficientnetv2_l as create_model

Image.MAX_IMAGE_PIXELS = 2300000000

def main(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "l"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         # transforms.Grayscale(num_output_channels=3),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    # img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    # plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    # 改json路径
    qt_proj_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(qt_proj_dir, 'class_indices.json')
    # json_path = 'QtProject/class_indices.json'

    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = create_model(num_classes=3577).to(device)
    # load model weights
    # 改路径
    model_weight_path = os.path.join(qt_proj_dir, 'keep_weight', 'model-99.pth')
    # model_weight_path = "QtProject/keep_weight/model-99.pth"

    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    prob = float(predict[predict_cla].numpy())

    cls_name = class_indict[str(predict_cla)]

    return cls_name, prob


if __name__ == '__main__':
    path = 'E:\BaiduNetdiskDownload\stamp\stamp_classify_adjust'
    # run(path)
