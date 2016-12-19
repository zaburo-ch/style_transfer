from scipy.misc import imread, imsave
import argparse
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
"""

content_image = imread('../content_input/girl.jpg')
style_image = imread('../style_input/kizan-daruma_cropped.jpg')
result_image = style_transfer(content_image, style_image)
imsave("daruma-girl.png", result_image)