from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
from tqdm import tqdm

def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    if is_void:
        return 1.0
    else:
        return iou

def compute_IoU_cIoU(input_json, gti_annotations):
    # Ground truth annotations
    coco_gti = COCO(gti_annotations)

    # Predictions annotations
    #submission_file = json.loads(open(input_json).read())
    # coco = COCO(gti_annotations)
    # coco = coco.loadRes(submission_file)
    with open(input_json, 'r') as f:
        result = json.load(f)
    id_result = {}
    for item in result:
        if item['image_id'] in id_result.keys():
            id_result[item['image_id']].append(item)
        else:
            id_result[item['image_id']] = [item]

    image_ids = coco_gti.getImgIds(catIds=coco_gti.getCatIds())
    bar = tqdm(image_ids)

    list_iou = []
    list_ciou = []
    for image_id in bar:

        img = coco_gti.loadImgs(image_id)[0]
        # 
        # annotation_ids = coco.getAnnIds(imgIds=img['id'])
        # annotations = coco.loadAnns(annotation_ids)
        if image_id not in id_result.keys():
            continue
        annotations = id_result[image_id]
        N = 0
        for _idx, annotation in enumerate(annotations):
            score = annotation['score']
            if score < 0.12:
                continue
            rle = cocomask.frPyObjects([annotation['polys']], 300, 300)
            #rle = annotation['segmentation']
            m = cocomask.decode(rle)
            if _idx == 0:
                mask = m.reshape((300, 300))
                #N = len(annotation['segmentation'][0]) // 2
                N = len(annotation['polys']) // 2
            else:
                mask = mask + m.reshape((300, 300))
                #N = N + len(annotation['segmentation'][0]) // 2
                N = N + len(annotation['polys']) // 2

        mask = mask != 0


        annotation_ids = coco_gti.getAnnIds(imgIds=img['id'])
        annotations = coco_gti.loadAnns(annotation_ids)
        N_GT = 0
        for _idx, annotation in enumerate(annotations):
            rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            m = cocomask.decode(rle)
            if _idx == 0:
                mask_gti = m.reshape((300, 300))
                N_GT = len(annotation['segmentation'][0]) // 2 - 1
            else:
                mask_gti = mask_gti + m.reshape((300, 300))
                N_GT = N_GT + len(annotation['segmentation'][0]) // 2

        mask_gti = mask_gti != 0

        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
        iou = calc_IoU(mask, mask_gti)
        list_iou.append(iou)
        list_ciou.append(iou * ps)

        bar.set_description("iou: %2.4f, c-iou: %2.4f" % (np.mean(list_iou), np.mean(list_ciou)))
        bar.refresh()

    print("Done!")
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))



if __name__ == "__main__":
    compute_IoU_cIoU(input_json="/root/zt/projects/mmdet_corner/line_seg/work_dirs/polygonal_lineformer_swinL_crowdAI_results.segm.json",
                     gti_annotations="/root/zt/projects/mmdet_corner/line_seg/data/CrowdAI/val/annotation.json")
