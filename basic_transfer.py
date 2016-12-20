from scipy.misc import imread, imsave
import argparse
from my_style_transfer import style_transfer


parser = argparse.ArgumentParser(description='Style Transfer with TensorFlow.')
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

content_image = imread(content_image_path)
style_image = imread(style_image_path)
result_image = style_transfer(content_image,
                              style_image,
                              content_weight=content_weight,
                              style_weight=style_weight,
                              init_image_type=init_image_type)
result_image = style_transfer(content_image, style_image)
imsave(result_prefix + "result_image.png", result_image)
