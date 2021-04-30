import argparse
import torch
import os
import cv2
import onnxruntime
import numpy as np

# /results/jxli/concepts/weights/yolov4_-1_3_608_608_dynamic.onnx
# IN  >>> /results/jxli/ActivityNet/data/{test30cut, val30cut1, val30cut2}
# OUT >>> /results/jxli/ActivityNet/data/{test30cutyolo, val30cut1yolo, val30cut2yolo}
# run -p gpu_long --mem=10000 --gres=gpu:1 --pty bash
# p yolo_runner.py ../concepts/weights/yolov4_-1_3_608_608_dynamic.onnx /results/jxli/ActivityNet/data/train30cut /results/jxli/ActivityNet/data/train30cutyolo 32
# p yolo_runner.py ../concepts/weights/yolov4_-1_3_608_608_dynamic.onnx /results/jxli/ActivityNet/data/train30cut /results/jxli/ActivityNet/data/train30cutyolo 32
# p yolo_runner.py ../concepts/weights/yolov4_-1_3_608_608_dynamic.onnx /results/jxli/ActivityNet/data/train30cut /results/jxli/ActivityNet/data/train30cutyolo 32

parser = argparse.ArgumentParser(description="Script to run YOLOv4 on videos")
parser.add_argument('onnxpath', type=str, help='Onnx yolov4 file')
# has format v_#########_30_i.mp4
parser.add_argument('indir', type=str,
                    help='path to the already cut 30 FPS videos')
parser.add_argument('outdir', type=str, help='path to place YOLO detections')
parser.add_argument('batchsize', type=str, help='YOLO batch size')
parser.add_argument('worker', type=str, help='workeridx, [0, nworkers)')
parser.add_argument('nworkers', type=str, help='number of workers')

ARGS = parser.parse_args()
onnxpath, indir, outdir, batchsize = ARGS.onnxpath, ARGS.indir, ARGS.outdir, int(ARGS.batchsize)
workeridx, nworkers = int(ARGS.workeridx), int(ARGS.nworkers)

print(onnxpath, indir, outdir, batchsize)

session = onnxruntime.InferenceSession(onnxpath)
_, _, IN_IMAGE_H, IN_IMAGE_W = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name


def inferbatch(batch):
    """Run YOLO on a list of images

    Args:
        batch (list): pylist of cv2 images
    """
    global session, IN_IMAGE_H, IN_IMAGE_W, input_name
    batch = [cv2.cvtColor(cv2.resize(img, (IN_IMAGE_W, IN_IMAGE_H),
                                     interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB) for img in batch]
    batch = np.stack(batch)
    batch = np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)
    batch /= 255.0
    outputs = session.run(None, {input_name: batch})
    HOLD = post_processing(0.4, 0.6, outputs)
    return HOLD


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(conf_thresh, nms_thresh, output):
    box_array, confs = output
    # [batch, num, 1, 4], [batch, num, num_classes]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2],
                                   ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)

    return bboxes_batch


for root, dirs, files in os.walk(indir):
    nfiles = len(f)
    numperworker = nfiles // nworkers
    startidx = workeridx * numperworker
    endidx = startidx + numperworker if workeridx != (nworkers - 1) else None
    for vididx, filename in enumerate(sorted(files)[startidx:endidx], start=startidx):
        # filename has format v_12345678901_30_i.mp4
        print(f"Processing video ID: {filename[2:13]}, segment {filename[17:-4]}")
        print(f"WorkerIDX: {workeridx}/{nworkers}, vididx: {vididx}, assignment:[{startidx}, {endidx})")
        vidpath = os.path.join(root, filename)
        if vidpath[-4:] != ".mp4":
            print(f"File {vidpath} is not video. Skipping...")
            continue  # skip non-videos

        cap = cv2.VideoCapture(vidpath)
        out = []
        batch = []
        n_frames = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            n_frames += 1
            batch.append(frame)
            if len(batch) == batchsize:
                out += inferbatch(batch)
                batch = []

        if len(batch) != 0:
            out += inferbatch(batch)

        assert n_frames == len(out)
        
        _outpath = os.path.join(outdir, f"{filename[:-4]}_yolo4.txt")
        print(f"Writing results of {n_frames} frames to {_outpath}")
        with open(_outpath, 'w') as f:
            for frame in out:
                f.write(str(frame) + "\n")
