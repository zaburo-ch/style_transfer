import numpy as np
import argparse
import os
from scipy.misc import imread, imsave
from my_style_transfer import style_transfer


parser = argparse.ArgumentParser(description='Luminance-only transfer with TensorFlow.')
parser.add_argument('content_image_path', type=str,
                    help='Path to the content image.')
parser.add_argument('style_image_path', type=str,
                    help='Path to the style image.')
parser.add_argument('--result_prefix', type=str, default='result/',
                    help='Prefix for the saved results. (default: %(default)s)')
parser.add_argument('--content_weight', type=float, default=1e0,
                    help='Content weight in the loss function. (default: %(default)s)')
parser.add_argument('--style_weight', type=float, default=1e-1,
                    help='Style weight in the loss function. (default: %(default)s)')
parser.add_argument('--init_image_type', type=str, default='content',
                    help='Type of the initialization. (default: %(default)s)')

args = parser.parse_args()
content_image_path = args.content_image_path
style_image_path = args.style_image_path
result_prefix = args.result_prefix
content_weight = args.content_weight
style_weight = args.style_weight
init_image_type = args.init_image_type


# util functions

def save_image(img, name):
    img = np.clip(img, 0, 255).astype(np.uint8)
    imsave(result_prefix + name, img)


def transform(W, img):
    shape = img.shape
    img = img.reshape((-1, shape[2])).dot(W.T)
    img = img.reshape(shape)
    return img


def ascertain_dir():
    dirname = os.path.dirname(result_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# functions for interconversion between RGB and YIQ
# https://en.wikipedia.org/wiki/YIQ

def rgb_to_yiq(img):
    W = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.274, -0.322],
                  [0.211, -0.523, 0.312]],
                 dtype=np.float32)
    return transform(W, img)


def yiq_to_rgb(img):
    W = np.array([[1, 0.956, 0.621],
                  [1, -0.272, -0.647],
                  [1, -1.106, 1.703]],
                 dtype=np.float32)
    return transform(W, img)


# load images
content_image = imread(content_image_path)
style_image = imread(style_image_path)

# extract the luminance channel of content image
content_yiq = rgb_to_yiq(content_image)
content_y, content_iq = content_yiq[:, :, 0:1], content_yiq[:, :, 1:]
content_image = np.concatenate((content_y, np.zeros_like(content_iq)), axis=2)
content_image = yiq_to_rgb(content_image)

# extract the luminance channel of style image and match the histogram
style_yiq = rgb_to_yiq(style_image)
style_y, style_iq = style_yiq[:, :, 0:1], style_yiq[:, :, 1:]
style_y = (style_y - np.mean(style_y)) * np.std(style_y) / np.std(content_y) + np.mean(content_y)
style_image = np.concatenate((style_y, np.zeros_like(style_iq)), axis=2)
style_image = yiq_to_rgb(style_image)

# apply style transfer
result_raw = style_transfer(content_image,
                            style_image,
                            content_weight=content_weight,
                            style_weight=style_weight,
                            init_image_type=init_image_type)
result_yiq = rgb_to_yiq(result_raw)
result_image = np.concatenate((result_yiq[:, :, 0:1], content_iq), axis=2)
result_image = yiq_to_rgb(result_image)

# save as RGB image
ascertain_dir()
save_image(content_image, "content_image_y00.png")
save_image(yiq_to_rgb(np.concatenate((127.5 * np.ones_like(content_y), content_iq), axis=2)),
           "content_image_0iq.png")
save_image(yiq_to_rgb(np.concatenate((content_y, content_iq), axis=2)),
           "content_image_yiq.png")
save_image(style_image, "style_image_y00.png")

save_image(result_raw, "result_raw.png")
save_image(yiq_to_rgb(np.concatenate((result_yiq[:, :, 0:1], np.zeros_like(content_iq)), axis=2)), "result_y00.png")
save_image(result_image, "result_image.png")
