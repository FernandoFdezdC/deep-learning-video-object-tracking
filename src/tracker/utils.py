#########################################
# Still ugly file with helper functions #
#########################################

import os
import random
from collections import defaultdict
from os import path as osp

import cv2
import matplotlib
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import torch
from cycler import cycler as cy
from scipy.interpolate import interp1d
from torchvision.transforms import functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc

    
colors = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon',
    'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
    'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue',
    'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey',
    'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon',
    'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue',
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon',
    'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue',
    'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle',
    'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen'
]

def plot_sequence(tracks, db, first_n_frames=None):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
    """

    # print("[*] Plotting whole sequence to {}".format(output_dir))

    # if not osp.exists(output_dir):
    # 	os.makedirs(output_dir)

    # infinite color loop
    cyl = cy('ec', colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        img = v['img'].mul(255).permute(1, 2, 0).byte().numpy()
        width, height, _ = img.shape

        dpi = 96
        fig, ax = plt.subplots(1, dpi=dpi)
        fig.set_size_inches(width / dpi, height / dpi)
        ax.set_axis_off()
        ax.imshow(img)

        for j, t in tracks.items():
            if i in t.keys():
                t_i = t[i]
                ax.add_patch(
                    plt.Rectangle(
                        (t_i[0], t_i[1]),
                        t_i[2] - t_i[0],
                        t_i[3] - t_i[1],
                        fill=False,
                        linewidth=1.0, **styles[j]
                    ))

                ax.annotate(j, (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                            color=styles[j]['ec'], weight='bold', fontsize=6, ha='center', va='center')

        plt.axis('off')
        # plt.tight_layout()
        plt.show()
        # plt.savefig(im_output, dpi=100)
        # plt.close()

        if first_n_frames is not None and first_n_frames == i:
            break


def get_mot_accum(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for i, data in enumerate(seq):
        gt = data['gt']
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack((gt_boxes[:, 0],
                                 gt_boxes[:, 1],
                                 gt_boxes[:, 2] - gt_boxes[:, 0],
                                 gt_boxes[:, 3] - gt_boxes[:, 1]),
                                axis=1)
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack((track_boxes[:, 0],
                                    track_boxes[:, 1],
                                    track_boxes[:, 2] - track_boxes[:, 0],
                                    track_boxes[:, 3] - track_boxes[:, 1]),
                                    axis=1)
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(
            gt_ids,
            track_ids,
            distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall)

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(str_summary)


def evaluate_obj_detect(model, data_loader):
    model.eval()
    device = list(model.parameters())[0].device
    results = {}
    for imgs, targets in tqdm(data_loader):
        imgs = [img.to(device) for img in imgs]

        with torch.no_grad():
            preds = model(imgs)

        for pred, target in zip(preds, targets):
            results[target['image_id'].item()] = {'boxes': pred['boxes'].cpu(),
                                                  'scores': pred['scores'].cpu()}

    data_loader.dataset.print_eval(results)


def obj_detect_transforms(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def plot_confusion_matrix(y_pred_proba, y_truth, threshold=0.5):
    # Función que plotea la matriz de confusión dado un cierto umbral
    
    # Convertir las probabilidades en etiquetas de clase (0 o 1) usando el threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculamos matriz de confusión
    conf_matrix = confusion_matrix(y_truth, y_pred)
    
    plt.figure(figsize=(6, 6))
    # Matriz de confusión para conjunto de entrenamiento
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.coolwarm_r)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j],
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='white', size=20)
    plt.title('Matriz de Confusión\nThreshold='+str(threshold), size=20)
    plt.colorbar(shrink=0.6)  # Reducir el tamaño de la barra de color
    plt.xticks(np.arange(2), labels=['0', '1'])
    plt.yticks(np.arange(2), labels=['0', '1'])
    plt.xlabel('Clase Predicha', size=18)
    plt.ylabel('Clase Real', size=18)

    # Ajustar el diseño
    plt.tight_layout()

    # Mostrar la gráfica
    plt.show()

def calcular_f1_max(y_truth, y_proba):
    # precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_truth, y_proba, pos_label=1)
    
    # Calcular F1 máximo de todos los umbrales posibles
    # F1 es la media armónica entre precision y recall
    denominador = precision + recall
    denominador[denominador == 0] = 1.0  # Evitar división por cero
    f1_scores = 2 * (precision * recall) / denominador
    f1_scores = np.nan_to_num(f1_scores)  # Reemplazar NaN con cero
    max_f1_index = np.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_index]
    threshold_max_f1 = thresholds[max_f1_index]
    return max_f1, threshold_max_f1
    
def plot_precision_recall(y_pred_proba, y_truth):
    # Función que plotea la curva precision-recall

    # Calcular F1 máximo
    max_f1, max_f1_threshold = calcular_f1_max(y_truth, y_pred_proba)
    print(f"F1 máximo: \033[1m{max_f1:.4f}\033[0m")  # \033[1m para negrita, \033[0m para resetear el estilo
    print(f"Umbral óptimo: \033[1m{max_f1_threshold:.8f}\033[0m")
    
    precision, recall, _ = precision_recall_curve(y_truth, y_pred_proba, pos_label=1)
    
    # Calcular el área bajo la curva
    pr_auc = auc(recall, precision)

    # Calcular la curva de no skill
    no_skill = len(y_truth[y_truth==1]) / len(y_truth)
    
    # Graficar la curva precision-recall
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color='darkorange', marker='.', markersize=4, label=f'Curva Precision-Recall (AUC = {pr_auc:.4f})')
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.xlabel('Recall (Positive label: 1)', size=18)
    plt.ylabel('Precision (Positive label: 1)', size=18)
    plt.title('Precision-Recall Curve', size=20)
    plt.legend(loc='lower right')
    plt.show()

    # Plot best confusion matrix:
    plot_confusion_matrix(y_pred_proba, y_truth, threshold=max_f1_threshold)
    
    return max_f1_threshold