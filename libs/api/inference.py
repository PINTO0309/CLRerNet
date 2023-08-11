# modified based on:
# https://github.com/open-mmlab/mmdetection/blob/v2.28.0/mmdet/apis/inference.py
# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import torch
from mmcv.parallel import collate, scatter

from libs.datasets.pipelines import Compose
from libs.datasets.metrics.culane_metric import interp


def inference_one_image(model, img_path):
    """Inference on an image with the detector.
    Args:
        model (nn.Module): The loaded detector.
        img_path (str): Image path.
    Returns:
        img (np.ndarray): Image data with shape (width, height, channel).
        preds (List[np.ndarray]): Detected lanes.
    """
    img = cv2.imread(img_path)
    ori_shape = img.shape
    data = dict(
        filename=img_path,
        sub_img_name=None,
        img=img,
        gt_points=[],
        id_classes=[],
        id_instances=[],
        img_shape=ori_shape,
        ori_shape=ori_shape,
    )

    cfg = model.cfg
    model.bbox_head.test_cfg.as_lanes = False
    device = next(model.parameters()).device  # model device

    test_pipeline = Compose(cfg.data.test.pipeline)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    data['img_metas'] = data['img_metas'].data[0]
    data['img'] = data['img'].data[0]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        """
        'img_metas':
            [
                {
                    'filename': 'demo/demo.jpg',
                    'sub_img_name': None,
                    'ori_shape': (590, 1640, 3),
                    'img_shape': (320, 800, 3),
                    'img_norm_cfg': {
                        'mean': array([0., 0., 0.], ...e=float32),
                        'std': array([255., 255., 255...e=float32),
                        'to_rgb': False
                    }
                }
            ]
        'img': torch.Size([1, 3, 320, 800])
            tensor([[[[0.2196, 0.2196, 0.2196,  ..., 0.2510, 0.2745, 0.2980],
        """
        # results = model(return_loss=False, rescale=True, **data)

        import onnx
        from onnxsim import simplify
        RESOLUTION = [
            [320,800],
            # [192,320],
            # [192,416],
            # [192,640],
            # [192,800],
            # [256,320],
            # [256,416],
            # [256,640],
            # [256,800],
            # [256,960],
            # [288,480],
            # [288,640],
            # [288,800],
            # [288,960],
            # [288,1280],
            # [320,320],
            # [384,480],
            # [384,640],
            # [384,800],
            # [384,960],
            # [384,1280],
            # [416,416],
            # [480,640],
            # [480,800],
            # [480,960],
            # [480,1280],
            # [512,512],
            # [544,800],
            # [544,960],
            # [544,1280],
            # [640,640],
        ]
        model.cpu()
        MODEL = f'clrernet_no_nms_no_predictions_to_lanes'
        for H, W in RESOLUTION:
            x = torch.nn.functional.interpolate(data['img'], size=(H,W)).cpu()
            onnx_file = f"{MODEL}_{x.shape[2]}x{x.shape[3]}.onnx"
            torch.onnx.export(
                model,
                args=(x),
                f=onnx_file,
                opset_version=16,
                input_names=['input'],
                output_names=[
                    'xs',
                    'anchor_params',
                    'lengths',
                    'scores',
                ],
                dynamic_axes={
                    'xs' : {0: 'N'},
                    'anchor_params' : {0: 'N'},
                    'lengths' : {0: 'N'},
                    'scores' : {0: 'N'},
                }
            )
            model_onnx1 = onnx.load(onnx_file)
            model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
            onnx.save(model_onnx1, onnx_file)

            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)
            model_onnx2 = onnx.load(onnx_file)
            model_simp, check = simplify(model_onnx2)
            onnx.save(model_simp, onnx_file)

        onnx_file = f"{MODEL}_HxW.onnx"
        torch.onnx.export(
            model,
            args=(x),
            f=onnx_file,
            opset_version=16,
            input_names=['input'],
                output_names=[
                    'xs',
                    'anchor_params',
                    'lengths',
                    'scores',
                ],
            dynamic_axes={
                'input' : {2: 'height', 3: 'width'},
                'xs' : {0: 'N'},
                'anchor_params' : {0: 'N'},
                'lengths' : {0: 'N'},
                'scores' : {0: 'N'},
            }
        )
        model_onnx1 = onnx.load(onnx_file)
        model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
        onnx.save(model_onnx1, onnx_file)

        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)
        model_onnx2 = onnx.load(onnx_file)
        model_simp, check = simplify(model_onnx2)
        onnx.save(model_simp, onnx_file)

        import sys
        sys.exit(0)



    lanes = results[0]['result']['lanes']
    preds = get_prediction(lanes, ori_shape[0], ori_shape[1])

    return img, preds


def get_prediction(lanes, ori_h, ori_w):
    preds = []
    for lane in lanes:
        lane = lane.cpu().numpy()
        xs = lane[:, 0]
        ys = lane[:, 1]
        valid_mask = (xs >= 0) & (xs < 1)
        xs = xs * ori_w
        lane_xs = xs[valid_mask]
        lane_ys = ys[valid_mask] * ori_h
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
        pred = [(x, y) for x, y in zip(lane_xs, lane_ys)]
        interp_pred = interp(pred, n=5)
        preds.append(interp_pred)
    return preds
