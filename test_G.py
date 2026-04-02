import os
import argparse  # REFACTOR: added to replace hardcoded paths and CUDA device

import ops


# REFACTOR: parse_args() added — was no argparse; all paths were hardcoded Windows drives
def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained RegiStain Generator')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing target images, e.g. /data/BCI/testB/')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .h5 Generator checkpoint, e.g. model_G_iter=87700.h5')
    parser.add_argument('--output_dir', required=True,
                        help='Directory where output PNGs will be saved')
    parser.add_argument('--gpu', default='0',
                        help='CUDA_VISIBLE_DEVICES value (default: 0)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Spatial size of input images in pixels (default: 256; original autopsy slides used 2048)')
    parser.add_argument('--is_mat', action='store_true',
                        help='Use .mat autofluorescence loading instead of RGB image loading')
    return parser.parse_args()


args = parse_args()

# REFACTOR: was hardcoded os.environ["CUDA_VISIBLE_DEVICES"] = "0" at module level;
# moved after parse_args() so --gpu arg is applied before TensorFlow reads the env var
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import glob
from configobj import ConfigObj
from models import att_unet_2d
from tqdm import tqdm
from losses import *
import scipy.io as sio
import matplotlib.pyplot as plt
import batch_utils
from batch_utils import ImageTransformationBatchLoader_Testing


def init_parameters():
    tc, vc = ConfigObj(), ConfigObj()

    # REFACTOR: were hardcoded 'L:\\...' Windows backslash paths; now built from --data_dir and --is_mat args
    ext = 'mat' if args.is_mat else 'png'
    tc.image_path = os.path.join(args.data_dir, f'*.{ext}')
    vc.image_path = tc.image_path

    if args.is_mat:
        # original behavior: .mat files with 'input'/'target' keys in sibling folders
        def convert_inp_path_from_target(inp_path: str):
            return inp_path.replace('target', 'input')
    else:
        # REFACTOR: RGB A/B folder layout for BCI/MIST-HER2 (e.g. testB/*.png → testA/*.png)
        def convert_inp_path_from_target(inp_path: str):
            d, f = os.path.split(inp_path)
            parent = os.path.dirname(d)
            folder = os.path.basename(d).replace('B', 'A')  # testB→testA
            return os.path.join(parent, folder, f)

    tc.convert_inp_path_from_target = convert_inp_path_from_target
    vc.convert_inp_path_from_target = convert_inp_path_from_target

    # REFACTOR: was hardcoded True; now driven by --is_mat flag (default False = RGB mode)
    tc.is_mat, vc.is_mat = args.is_mat, args.is_mat
    tc.data_inpnorm, vc.data_inpnorm = False, False  # True for normalizing input images

    tc.channel_start_index, vc.channel_start_index = 0, 0
    tc.channel_end_index, vc.channel_end_index = 2, 2  # exclusive

    tc.is_training, vc.is_training = True, False
    # REFACTOR: was hardcoded 2048 (for full autopsy slides); now from --image_size arg (default 256)
    tc.image_size, vc.image_size = args.image_size, args.image_size
    tc.num_slices, vc.num_slices = 2, 2
    tc.label_channels, vc.label_channels = 3, 3

    assert tc.channel_end_index - tc.channel_start_index == tc.num_slices
    assert vc.channel_end_index - vc.channel_start_index == vc.num_slices
    tc.n_channels, vc.n_channels = 32, 32

    tc.batch_size, vc.batch_size = 1, 1
    tc.n_threads, vc.n_threads = 2, 2
    tc.q_limit, vc.q_limit = 10, 10
    tc.n_shuffle_epoch, vc.n_shuffle_epoch = 1, 1  # for the batchloader
    tc.data_inpnorm, vc.data_inpnorm = 'norm_by_mean_std', 'norm_by_mean_std'

    return tc, vc


if __name__ == '__main__':
    # REFACTOR: were three hardcoded 'L:/...' Windows paths; now from --checkpoint and --output_dir args
    checkpoint_path = args.checkpoint
    output_path = args.output_dir
    tf.io.gfile.mkdir(output_path)

    # initialize architecture and load weights
    tc, vc = init_parameters()
    model_G = att_unet_2d((tc.image_size, tc.image_size, tc.num_slices), n_labels=tc.label_channels, name='model_G',
                          filter_num=[tc.n_channels, tc.n_channels * 2, tc.n_channels * 4, tc.n_channels * 8, tc.n_channels * 16],
                          stack_num_down=3, stack_num_up=3, activation='LeakyReLU',
                          atten_activation='ReLU', attention='add',
                          output_activation=None, batch_norm=True, pool='ave', unpool='bilinear')
    model_G.load_weights(checkpoint_path)

    ssim_list = []
    psnr_list = []

    # _, _, test_images = batch_utils.Her2data_splitter(tc)
    test_images = glob.glob(vc.image_path)

    valid_bl = ImageTransformationBatchLoader_Testing(test_images, vc, vc.num_slices, is_testing=True,
                                              n_parallel_calls=vc.n_threads, q_limit=vc.q_limit,
                                              n_epoch=vc.n_shuffle_epoch)
    iterator_valid_bl = iter(valid_bl.dataset)

    # loop over batches

    print('valid images: ' + str(test_images[:2]))
    for i in tqdm(range(len(test_images) // tc.batch_size)):
        valid_x, valid_y = next(iterator_valid_bl)

        # with tf.device('/cpu:0'):
        with tf.device('/gpu:0'):
            valid_output = model_G(valid_x, training=False).numpy()

        for j in range(tc.batch_size):
            valid_output_temp = np.clip(valid_output[j], 0, 1)
            # REFACTOR: was tf.concat([valid_x[j,:,:,0:2], valid_x[j,:,:,3:4]], axis=-1)
            # channel index 3 does not exist for 2-channel input — simplified to use full input as-is
            valid_x_temp = valid_x[j]
            valid_x_temp = (valid_x_temp / tf.reduce_max(valid_x_temp)).numpy()
            valid_y_temp = valid_y.numpy() * 255
            valid_y_temp = valid_y_temp[j]
            valid_image_path = test_images[i * tc.batch_size + j]

            # REFACTOR: was valid_image_path.split('\\')[-1] — backslash split breaks on Linux
            cur_out_img_name = os.path.splitext(os.path.basename(valid_image_path))[0] + '.png'

            # REFACTOR: was valid_image_path.split('\\')[-3] — backslash split breaks on Linux
            cur_case_name = os.path.basename(os.path.dirname(os.path.dirname(valid_image_path)))
            plt.imsave(output_path + cur_case_name + '_' + cur_out_img_name,
                       valid_output_temp)
