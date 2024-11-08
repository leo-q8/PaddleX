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

import numpy as np

from ..results import KeypointResult
from ..utils.io import ImageReader
from .base import BasePipeline
from ...utils import logging


class KeypointDetectionPipeline(BasePipeline):
    """Keypoint Detection Pipeline"""

    entities = "human_keypoint_detection"

    def __init__(
        self,
        human_det_model,
        keypoint_det_model,
        human_det_batch_size=1,
        keypoint_det_batch_size=1,
        device=None,
        predictor_kwargs=None,
    ):
        super().__init__(device, predictor_kwargs)
        self._build_predictor(human_det_model, keypoint_det_model)
        self.set_predictor(
            human_det_batch_size=human_det_batch_size,
            keypoint_det_batch_size=keypoint_det_batch_size,
        )
        self._img_reader = ImageReader(backend="opencv")

    def _build_predictor(self, human_det_model, keypoint_det_model):
        self.human_detector = self._create(model=human_det_model)
        self.keypoint_detector = self._create(model=keypoint_det_model)
        affine_trans = self.keypoint_detector.components['TopDownEvalAffine']
        affine_trans.set_inputs({
            "img": "img", "center": "center", "scale": "scale"})
        self.keypoint_infer_size = affine_trans.trainsize
    
    def _box_xyxy2cs(self, bbox, padding=1.25):
        x1, y1, x2, y2 = bbox[:4]
        center = np.array([x1 + x2, y1 + y2]) * 0.5

        # reshape bbox to fixed aspect ratio
        aspect_ratio = self.keypoint_infer_size[0] / self.keypoint_infer_size[1]
        w, h = x2 - x1, y2 - y1
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h]) * padding

        return center, scale

    def set_predictor(
        self, human_det_batch_size=None, keypoint_det_batch_size=None, device=None
    ):
        if human_det_batch_size:
            self.human_detector.set_predictor(batch_size=human_det_batch_size)
        if keypoint_det_batch_size:
            self.keypoint_detector.set_predictor(batch_size=keypoint_det_batch_size)
        if device:
            self.human_detector.set_predictor(device=device)
            self.keypoint_detector.set_predictor(device=device)

    def predict(self, input, **kwargs):
        self.set_predictor(**kwargs)
        for det_res in self.human_detector(input):
            keypoint_res = self.get_keypoint_result(det_res)
            yield self.get_final_result(det_res, keypoint_res)

    def get_keypoint_result(self, det_res):
        img = self._img_reader.read(det_res["input_path"])
        input_list = [{
                "input": {"img": img, "center": center, "scale": scale}
            } for center, scale in map(
            lambda x: self._box_xyxy2cs(x["coordinate"]), det_res["boxes"])]
        all_keypoint_res = list(self.keypoint_detector(input_list))
        return all_keypoint_res
    
    def get_final_result(self, det_res, keypoint_res):
        single_img_res = {"input_path": det_res["input_path"], "boxes": []}
        for i, obj in enumerate(det_res["boxes"]):
            single_img_res["boxes"].append({
                "det_score": obj["score"],
                "coordinate": obj["coordinate"],
                "keypoints": keypoint_res[i]["keypoints"],
            })
        return KeypointResult(single_img_res)
 