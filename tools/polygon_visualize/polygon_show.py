import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import os
import json
import cv2

class Visualizer:

    save_dir = 'work_dirs/polygons_demo'

    def visualize_ex(self, img, polygons, scores, img_name, thr=0.3):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(img)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(polygons)):
            if scores[i] < thr:
                continue
            color = next(colors).tolist()
            poly = np.array(polygons[i]).reshape(-1, 2)
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, lw=2)
        plt.savefig(fname=os.path.join(self.save_dir, img_name.split('.')[0] + '.png'))
        #plt.savefig(fname=os.path.join(self.save_dir, img_name.split('.')[0] + '.png'), dpi=300)
        #plt.show()
        plt.close()

def run_visualize(img_dir, json_dir, thr=0.3):
    import tqdm
    visualizer = Visualizer()
    jsons = os.listdir(json_dir)
    img_type = os.listdir(img_dir)[0].split('.')[-1]

    for json_name in tqdm.tqdm(jsons):
        img_name = json_name[:-4] + img_type
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
        img = np.array(cv2.imread(img_path))[:, :, ::-1]
        with open(os.path.join(json_dir, json_name), 'r') as f:
            data = json.load(f)
        visualizer.visualize_ex(img, data['polygons'], data['scores'], img_name, thr=thr)

if __name__ == "__main__":
    run_visualize('data/whu/val', 'work_dirs/json_pred', 0.3)
    #run_visualize('data/CrowdAI/val/images/', 'work_dirs/json_pred', 0.3)
    #run_visualize('data/whu-mix/test/image/', 'work_dirs/json_pred', 0.3)
    #run_visualize('data/whu-mix/test2/image/', 'work_dirs/json_pred', 0.3)
