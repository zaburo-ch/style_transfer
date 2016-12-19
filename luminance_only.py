import numpy as np
import argparse
import os
from scipy.misc import imread, imsave
from my_style_transfer import style_transfer


"""
parser = argparse.ArgumentParser(description='Luminance-only transfer with Keras.')
parser.add_argument('content_image_path', metavar='content', type=str,
                    help='Path to the content image.')
parser.add_argument('style_image_path', metavar='style', type=str,
                    help='Path to the style image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    default=None, help='Prefix for the saved results.')
parser.add_argument('content_weight', metavar='w_content', type=float,
                    default=0.025, help='Content weight of loss function.')
parser.add_argument('style_weight', metavar='w_style', type=float,
                    default=1., help='Style weight of loss function.')
parser.add_argument('total_variation_weight', metavar='w_total', type=float,
                    default=1., help='Total variation weight of loss function.')

args = parser.parse_args()
content_image_path = args.content_image_path
style_image_path = args.style_image_path
result_prefix = args.result_prefix
content_weight = args.content_weight
style_weight = args.style_weight
total_variation_weight = args.total_variation_weight
"""

content_image_path = "../content_input/512_DSC_0489.jpg"
style_image_path = "../style_input/starry-night.jpg"
result_prefix = "result/gogh-0489_init-content_avg/"
content_weight = 1e0
style_weight = 1e-1
init_image_type = 'content'

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
                            init_image_type='content')
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
