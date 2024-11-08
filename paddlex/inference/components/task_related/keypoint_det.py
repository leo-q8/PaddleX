# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import cv2
import numpy as np

from ..base import BaseComponent


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def translate_to_ori_images(keypoint_result, batch_records):
    kpts = keypoint_result['keypoint']
    scores = keypoint_result['score']
    kpts[..., 0] += batch_records[:, 0:1]
    kpts[..., 1] += batch_records[:, 1:2]
    return kpts, scores


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt

def get_affine_transform(center,
                         input_size,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(output_size) == 2
    assert len(shift) == 2
    if not isinstance(input_size, (np.ndarray, list)):
        input_size = np.array([input_size, input_size], dtype=np.float32)
    scale_tmp = input_size

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """This code is based on 
        https://github.com/open-mmlab/mmpose/blob/master/mmpose/core/post_processing/post_transforms.py

        Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = np.cos(theta) * scale_x
    matrix[0, 1] = -np.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (
        -0.5 * size_input[0] * np.cos(theta) + 0.5 * size_input[1] *
        np.sin(theta) + 0.5 * size_target[0])
    matrix[1, 0] = np.sin(theta) * scale_y
    matrix[1, 1] = np.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (
        -0.5 * size_input[0] * np.sin(theta) - 0.5 * size_input[1] *
        np.cos(theta) + 0.5 * size_target[1])
    return matrix


class TopDownEvalAffine(BaseComponent):

    INPUT_KEYS = [["img", "center", "scale"],
                  ["img"]]
    OUTPUT_KEYS = ["img", "center", "scale"]
    DEAULT_INPUTS = {"img": "img", }
    DEAULT_OUTPUTS = {"img": "img", "center": "center", "scale": "scale"}

    """apply affine transform to image and coords
    
    Args:
        trainsize (list): [w, h], the standard size used to train
        use_udp (bool): whether to use Unbiased Data Processing.
        records(dict): the dict contained the image and coords
    
    Returns:
        records (dict): contain the image and coords after tranformed
    
    """
    
    def __init__(self, trainsize, use_udp=False):
        super().__init__()
        self.trainsize = trainsize
        self.use_udp = use_udp
    
    def apply(self, img, center=None, scale=None):
        rot = 0
        imshape = np.array(img.shape[:2][::-1])
        center = center if center is not None else imshape / 2.
        scale = scale if scale is not None else imshape
        if self.use_udp:
            trans = get_warp_matrix(
                rot, center * 2.0,
                [self.trainsize[0] - 1.0, self.trainsize[1] - 1.0], scale)
            img = cv2.warpAffine(
                img,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
        else:
            trans = get_affine_transform(center, scale, rot, self.trainsize)
            img = cv2.warpAffine(
                img,
                trans, (int(self.trainsize[0]), int(self.trainsize[1])),
                flags=cv2.INTER_LINEAR)
        return {
            "img": img,
            "center": center,
            "scale": scale
        } 
    

class KeypointPostProcess(BaseComponent):
    """Save Result Transform"""

    INPUT_KEYS = ["heatmap", "center", "scale"]
    OUTPUT_KEYS = ["keypoints", "score"]
    DEAULT_INPUTS = {"heatmap": "heatmap", "center": "center", "scale": "scale"}
    DEAULT_OUTPUTS = {"keypoints": "keypoints", "score": "score"}

    def __init__(self, use_dark=True):
        super().__init__()
        self.use_dark = use_dark

    def apply(self, heatmap, center, scale):
        """apply"""
        heatmap, center, scale = heatmap[None, ...], center[None, ...], scale[None, ...]
        preds, maxvals = self.get_final_preds(heatmap, center, scale)
        return {
            'keypoints': np.concatenate((preds, maxvals), axis=-1),
            'score': np.mean(maxvals, axis=1)
        }


    def get_final_preds(self, heatmaps, center, scale, kernelsize=3):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.

        Args:
            heatmaps (numpy.ndarray): The predicted heatmaps
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 1]), the maximum confidence of the keypoints
        """

        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        if self.use_dark:
            coords = self.dark_postprocess(heatmaps, coords, kernelsize)
        else:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array([
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ])
                        coords[n][p] += np.sign(diff) * .25
        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], center[i], scale[i],
                                       [heatmap_width, heatmap_height])

        return preds, maxvals

    def get_max_preds(self, heatmaps):
        """get predictions from score maps

        Args:
            heatmaps: numpy.ndarray([batch_size, num_joints, height, width])

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        """
        assert isinstance(heatmaps,
                          np.ndarray), 'heatmaps should be numpy.ndarray'
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask

        return preds, maxvals

    def gaussian_blur(self, heatmap, kernel):
        border = (kernel - 1) // 2
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border:-border, border:-border] = heatmap[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmap[i, j] = dr[border:-border, border:-border].copy()
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])
        return heatmap

    def dark_parse(self, hm, coord):
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
                + hm[py-1][px-1])
            dyy = 0.25 * (
                hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord

    def dark_postprocess(self, hm, coords, kernelsize):
        """
        refer to https://github.com/ilovepose/DarkPose/lib/core/inference.py

        """
        hm = self.gaussian_blur(hm, kernelsize)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self.dark_parse(hm[n][p], coords[n][p])
        return coords
