import os
import argparse
import glob

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from skimage import io
from PIL import Image
from torchvision import transforms

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def save_output_nobg(image_name, pred, d_dir):
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()

    original_img = Image.open(image_name).convert('RGB')
    original_img = original_img.resize((predict_np.shape[1], predict_np.shape[0]), resample=Image.BILINEAR)

    # convert prediction to an image
    im = Image.fromarray((predict_np * 255).astype(np.uint8))

    mask = im.convert('L')
    empty = Image.new("RGBA", original_img.size)
    empty.paste(original_img, (0, 0), mask=mask)

    img_name = os.path.splitext(os.path.basename(image_name))[0]
    empty.save(os.path.join(d_dir, f'{img_name}_nobg.png'))

def batch_prediction(image_dir, prediction_dir, model, batch_size=4):
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)

    img_name_list = glob.glob(os.path.join(image_dir, '*'))
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=1)

    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image'].type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, _, _, _, _, _, _ = model(inputs_test)
        pred = normPRED(d1[:, 0, :, :])

        for i in range(batch_size):
            if i_test * batch_size + i < len(img_name_list):
                save_output_nobg(img_name_list[i_test * batch_size + i], pred[i], prediction_dir)

def load_model(model_name='u2net'):
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')
    if model_name == 'u2net':
        print("Loading U2NET...")
        net = U2NET(3,1)
    else:  # Assume 'u2netp'
        print("Loading U2NETP...")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    return net

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch Background Removal with U-2-NET")
    parser.add_argument('-i', "--image-dir", type=str, default="./test_data/test_images", help="Directory of images to process")
    parser.add_argument('-o', "--prediction-dir", type=str, default="./test_data/u2net_results-3", help="Directory to save predictions")
    parser.add_argument('-m', "--model-name", type=str, choices=['u2net', 'u2netp'], default='u2net', help="Model name: 'u2net' (default) or 'u2netp'")
    parser.add_argument('-b', "--batch-size", type=int, default=32, help="Batch size for processing images")
    args = parser.parse_args()

    model = load_model(args.model_name)
    batch_prediction(args.image_dir, args.prediction_dir, model, args.batch_size)

