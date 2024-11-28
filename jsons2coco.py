import numpy as np
import json
from tqdm import tqdm
import os
from pycocotools import mask as cocomask

jsons_dir = './work_dirs/json_pred'
json_names = os.listdir(jsons_dir)
json_names = [os.path.join(jsons_dir, name) for name in json_names]

gt_json_path = './data/CrowdAI/val/annotation.json'
with open(gt_json_path, 'r') as f:
    gt_json = json.load(f)
images = gt_json['images']
name_images = {}
for image in images:
    name_images[image['file_name']] = image

res = []

bar = tqdm(json_names)
for path in bar:
    with open(path, 'r') as f:
        image_polys = json.load(f)
    polys = image_polys['polygons']
    scores = image_polys['scores']
    image_name = path.split('/')[-1][:-4] + 'jpg'
    image_info = name_images[image_name]

    image_id = image_info['id']
    if len(polys) == 0:
        rle = cocomask.frPyObjects([], image_info['height'], image_info['width'])[0]
        rle['counts'] = str(rle['counts'])
        res.append({'image_id': image_id, 'score': score, "category_id": 100, 'polys': [],
                    'segmentation': rle})
    for poly, score in zip(polys, scores):
        rle = cocomask.frPyObjects([poly], image_info['height'], image_info['width'])[0]
        rle['counts'] = str(rle['counts'])
        res.append({'image_id': image_id, 'score': score, "category_id": 100, 'polys': poly,
                    'segmentation': rle})

with open('./work_dirs/polygonal_lineformer_swinL_crowdAI_results.segm.json', 'w') as f:
    json.dump(res, f)

