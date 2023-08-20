import os
import time
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from network_files import FasterRCNN
from backbone import resnet34_fpn_backbone
from draw_box_utils import draw_box
import re

def create_model(num_classes):
    backbone = resnet34_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)
    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=2)

    # load train weights
    train_weights = "./weight_files/best-model.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = './landslide.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    # load image
    image_path = "./predict_files"

    data_transform = transforms.Compose([transforms.ToTensor()])

    model.eval()
    with torch.no_grad():
        images = os.listdir(os.path.join(image_path, "images"))
        for index, image in enumerate(images):
            img_path = os.path.join(image_path, "images", image)
            original_img = Image.open(img_path)
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            predictions = model(img.to(device))[0]

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            path_mat1 = re.split(r'\.', image)
            path_mat = os.path.join(image_path, "results", path_mat1[0])

            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=2)
            plt.imshow(original_img)
            original_img.save(os.path.join(image_path, "results", image))
            #io.savemat(path_mat + '.mat', {'box': predict_boxes, 'scores': predict_scores})

        print()


if __name__ == '__main__':
    main()
