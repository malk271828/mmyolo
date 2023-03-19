import numpy as np
import torch
from collections import defaultdict
from collections.abc import Iterable
from rich import print
from typing import List, Dict, Tuple, Optional, Union

import torch
from pytorch_lightning import Trainer, seed_everything
from ensemble_boxes import weighted_boxes_fusion

# mmlab libraries
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.device import get_device
from mmyolo.registry import HOOKS
# from mmcv.runner.hooks import HOOKS, Hook

# anomalib
from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer
from anomalib.data.utils import (
    generate_output_image_filename,
    get_image_filenames,
    read_image,
)

def recursive_inspect(iterable: Union[Iterable, int], depth: int = 0) -> None:
    """Recursively inspects an iterable object and prints its type and length.

    Args:
        iterable (Union[Iterable, int]): The iterable object to inspect.
        depth (int, optional): The depth of the iterable object in the recursive call stack. Defaults to 0.

    Returns:
        None: This function does not return anything.

    """
    print(f"depth:{depth} type:{type(iterable)}")
    if isinstance(iterable, Iterable):
        print(f"len:{len(iterable)}")
        if iterable:
            recursive_inspect(iterable[0], depth + 1)
        else:
            print(iterable)
    else:
        print(iterable)

@HOOKS.register_module()
class EnsembleHook(Hook):
    """EnsembleHook.
        This hook is used for ensemble object detection results from multiple models.
        It loads an anomaly detection model using the provided configuration and weight files and applies Weighted Boxes Fusion
        (WBF) to merge bounding boxes from the multiple models' detection results.
    """

    def __init__(self,
            anomalib_config_file,
            anomalib_weight_file):
        """
        Parameters
        ----------
            anomalib_config_file (str): The path to the configuration file for the anomaly detection model.
            anomalib_weight_file (str): The path to the weight file for the anomaly detection model.
        """

        # load anomaly detection model
        self.inferencer = TorchInferencer(
            config=anomalib_config_file,
            model_source=anomalib_weight_file, device=get_device())

        # file output for visualization
        self.writeToFile = False
        self.visualizer = Visualizer(mode="simple", task="detection")
        self.output_dir = "output/"

    def apply_wbf(self,
                  result_all_models: Dict,
                  ori_shape: Optional[Tuple[int, int]] = None,
                  iou_thr: float = 0.6,
                  skip_box_thr: float = 0.0001,
                  verbose: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Weighted Boxes Fusion (WBF) to multiple detections from different models.

        Parameters
        ----------
        result_all_models : List[dict]
            A list of dictionaries containing model detection results. Each dictionary should have the following keys:
            'bboxes' : numpy.ndarray, shape (N, 4), the coordinates of the N bounding boxes
            'scores' : numpy.ndarray, shape (N,), the confidence scores of the N bounding boxes
            'labels' : numpy.ndarray, shape (N,), the class labels of the N bounding boxes

        ori_shape : Tuple[int, int], optional
            The original shape of the image, given as a tuple of height and width, by default None.
            The axis order is (height, width)

        iou_thr : float, optional
            The IoU threshold for WBF, by default 0.6

        skip_box_thr : float, optional
            The threshold for removing small boxes during WBF, by default 0.0001

        verbose : int, optional
            The level of verbosity, by default 0

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the following numpy arrays:
            - The coordinates of the merged bounding boxes, shape (N, 4)
            - The confidence scores of the merged bounding boxes, shape (N,)
            - The class labels of the merged bounding boxes, shape (N,)
        """
        if verbose > 0:
            print("[pink]apply_wbf() begin -------------------------[/pink]")
            if verbose > 1:
                for i in range(len(result_all_models['bboxes'])):
                    print(f"number of boxes{i}: {len(result_all_models['bboxes'][i])}")

        if ori_shape:
            ori_shape_array = np.array(ori_shape[::-1] * 2)
        else:
            ori_shape_array = np.array([800, 800, 800, 800])

        assert len(result_all_models) > 0, "List of model detection results cannot be empty"
        # assert all(['bboxes' in r and 'scores' in r and 'labels' in r for r in result_all_models]), \
        #     "Each dictionary in the list must contain 'bboxes', 'scores', and 'labels' keys"
        assert len(result_all_models['bboxes']) == len(result_all_models['scores']) == \
            len(result_all_models['labels']), "The length of 'bboxes', 'scores', and 'labels' must be the same"

        # Converting from an absolute coordinate system to a relative coordinate system
        for i in range(len(result_all_models['bboxes'])):
            result_all_models['bboxes'][i] = result_all_models['bboxes'][i] / ori_shape_array

        bboxes, scores, labels = weighted_boxes_fusion(
            boxes_list=result_all_models['bboxes'],
            scores_list=result_all_models['scores'],
            labels_list=result_all_models['labels'],
            weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        if verbose > 0:
            if verbose > 1:
                print(f"number of boxes: {len(labels)}")
            print("[pink]apply_wbf() end -------------------------[/pink]")

        return bboxes, scores, labels

    def after_test_iter(self,
            runner: Runner,
            batch_idx: int,
            data_batch: dict,
            outputs: list
        ) -> None:
        """Callback function called after each test iteration.

        Parameters
        ----------
            runner (Runner): The current runner.
            batch_idx (int): The current batch index.
            data_batch (dict): The current testing data batch.
            outputs (list): The model output for the current testing data batch.
        """
        print(f"[red] after_test_iter [/red] batch_idx:{batch_idx} \
            length of data_samples:{len(data_batch['data_samples'])}")

        # Perform inference with anomaly detection model
        filename = data_batch["data_samples"][0].metainfo["img_path"]
        image = read_image(filename)
        predictions = self.inferencer.predict(image=image)
        if self.writeToFile:
            output = self.visualizer.visualize_image(predictions)
            file_path = generate_output_image_filename(input_path=filename, output_path=self.output_dir)
            self.visualizer.save(file_path=file_path, image=output)
            print(f"output to file: {file_path}")

        # Construct list of results from all the models
        result_all_models = defaultdict(list)
        bboxes = outputs[0].pred_instances.bboxes
        result_all_models["bboxes"].append(outputs[0].pred_instances.bboxes.cpu())
        result_all_models["scores"].append(outputs[0].pred_instances.scores.cpu())
        result_all_models["labels"].append(outputs[0].pred_instances.labels.cpu())
        # TODO:
        # https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/deploy/inferencers/base_inferencer.py#L87
        if predictions.pred_boxes is not None:
            num_sample = len(predictions.pred_boxes)
            result_all_models["bboxes"].append(predictions.pred_boxes)
            result_all_models["scores"].append([predictions.pred_score] * num_sample)
            result_all_models["labels"].append([0] * num_sample)
        else:
            result_all_models["bboxes"].append([0])
            result_all_models["scores"].append([0])
            result_all_models["labels"].append([0])

        # Apply weighted boxes fusion
        bboxes, scores, labels = self.apply_wbf(result_all_models,
            ori_shape=outputs[0].metainfo["ori_shape"], verbose = 2)

        # assign fusion results to the passed mutable object
        # https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/data_element.md
        del outputs[0].pred_instances.bboxes
        del outputs[0].pred_instances.scores
        del outputs[0].pred_instances.labels
        outputs[0].pred_instances.bboxes = torch.Tensor(bboxes)
        outputs[0].pred_instances.scores= torch.Tensor(scores)
        outputs[0].pred_instances.labels = torch.Tensor(labels)
