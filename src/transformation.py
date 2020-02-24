#adding library
from torchvision import transforms
import numpy as np
import cv2
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia

def train_transformation(image, size):
    #convert image to grayscale and find croping position
    np_img = np.asarray(image)
    grayImg = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    h, w = grayImg.shape
    medianImg = cv2.medianBlur(grayImg, 11)
    threshold = medianImg.max()/8
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = (medianImg > threshold)*1.0
    one_mask = np.argwhere(mask == 1)
    max_pos = np.max(one_mask, axis=0)
    min_pos = np.min(one_mask, axis=0)
    top, bottom, left, right = min_pos[0], max_pos[0], min_pos[1], max_pos[1]
    np_img = np_img[top:bottom, left: right, :]
    # Transform image
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    train_seq = iaa.Sequential(
        [
            iaa.Resize({"height": size, "width": size}),
            sometimes(iaa.Fliplr(1)),
            sometimes(iaa.Flipud(1)),
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=["constant", "edge"],
                pad_cval=0
            )),
            sometimes(iaa.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.2, 0.2),
                rotate=(-45, 45),
                order=[0, 1],
                cval=0,
                mode=["constant", "edge"]
            )),
            iaa.SomeOf((0,4),[
                iaa.OneOf([
                    iaa.WithChannels(0, iaa.Add((5, 50))),
                    iaa.WithChannels(1, iaa.Add((5, 20))),
                    iaa.WithChannels(2, iaa.Add((5, 20))),
                ]),
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0.0, 2.0)),
                    iaa.MedianBlur(k=(3, 5)),
                    iaa.AverageBlur(k=(2, 4))
                ]),
                iaa.contrast.LinearContrast((0.5, 1.5)),
                iaa.Sharpen(alpha=(0.0, 0.5),lightness=1.0),
                iaa.Multiply((0.5, 1.5)),
            ], random_order=True)
        ], random_order=False
    )
    np_img = train_seq.augment_images(np_img.reshape(1,*np_img.shape))

    # Convert to Pillow image
    im = Image.fromarray(np_img[0])

    # Convert to Tensor
    convert_to_tensor = [transforms.ToTensor()]
#     if normalize:
#         convert_to_tensor.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    tensor_tranform = transforms.Compose(convert_to_tensor)

    return tensor_tranform(im)

def inference_transformation(image, size):
    """ Transform image for validation

    Parameters
    ----------
    image: PIL.Image
        image to transform
    size: int
        size to scale

    Returns
    -------

    """
    #convert image to grayscale and find croping position
    np_img = np.asarray(image)
    grayImg = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    h, w = grayImg.shape
    medianImg = cv2.medianBlur(grayImg, 11)
    threshold = medianImg.max()/8
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = (medianImg > threshold)*1.0
    one_mask = np.argwhere(mask == 1)
    max_pos = np.max(one_mask, axis=0)
    min_pos = np.min(one_mask, axis=0)
    top, bottom, left, right = min_pos[0], max_pos[0], min_pos[1], max_pos[1]
    np_img = np_img[top:bottom, left: right, :]
    # Transform image
    val_seq = iaa.Sequential(
        [
            iaa.Resize({"height": int(size), "width": int(size)}),
        ], random_order=False
    )
    np_img = val_seq.augment_image(np_img)

    # Convert to Pillow image
    im = Image.fromarray(np_img)

    # Convert to Tensor
    transform_list = []
#     normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    transform_list.append(transforms.ToTensor())
#     transform_list.append(normalize)
    tensor_tranform=transforms.Compose(transform_list)

    return tensor_tranform(im)
