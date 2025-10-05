# deep-learning-video-object-tracking

Training of a video object tracking deep learning model using different distance between objects in network space calculations: Intersection over Union. Then, association follows using Hungarian algorithm. Also includes managing of false detections. The object detection algorithm used has a Fast-RCNN architecture. The video dataset used is MOT16.

Output results can be seen on the following link: https://drive.google.com/drive/folders/1yoOIrEYScLiQIvP9vdsOJcGo9r-xAX25?usp=sharing.


---

## Introduction

This project focuses on studying and improving a deep learning model for multi-object tracking. The starting point is a base model composed of a **Fast-RCNN object detector** trained on the **MOT16** dataset, and a tracker that assigns current detections to previous ones using greedy Intersection over Union (IoU). The goal is to improve the **MOTA** score by optimizing aspects of the base model.

The proposed improvements involve using only high-confidence detections from the detector (tuning the optimal threshold to maximize the F1-score) and applying the **Hungarian algorithm** for data association, which significantly improves the evaluation metrics.

---

## Method

### Base Model

The base model defines an object detector class `FRCNN_FPN` inherited from `FasterRCNN` in **torchvision**, composed of:

- **GeneralizedRCNNTransform**: Preprocesses input images (normalization and resizing).  
- **BackBoneWithFPN**: CNN backbone with Feature Pyramid Network (FPN) for feature extraction.  
- **IntermediateLayerGetter**: Extracts feature maps from different layers.  
- **Bottleneck**: Reduces dimensions and increases channels in feature maps using convolutional layers, batch normalization, and ReLU activation.  
- **FeaturePyramidNetwork**: Handles detection of objects at multiple scales.  
- **RoIHeads**: Processes proposed regions of interest (RoIs).  
- **FastRCNNPredictor**: Predicts class and bounding box regression for RoIs.  
- **Conv2d**: PyTorch convolutional layer.  
- **LastLevelMaxPool**: Reduces spatial resolution at the last FPN level.

The architecture extracts features and creates multi-scale feature maps for Faster R-CNN object detection.

### Tracker

The tracker assigns detections to tracks frame by frame.  

- **`Tracker`** methods:
  - `__init__(self, obj_detect)`: Initialize with an object detector.
  - `reset(self, hard=True)`: Reset tracks, counters, and results.
  - `add(self, new_boxes, new_scores)`: Initialize new tracks.
  - `get_pos(self)`: Return positions of active tracks.
  - `data_association(self, boxes, score)`: Associate new detections to existing tracks using IoU.
  - `step(self, frame)`: Track a single frame.
  - `get_results(self)`: Return tracking results.

- **`Track`** attributes:
  - `id`: Track ID.
  - `box`: Bounding box.
  - `score`: Detector confidence.

- **`TrackerIoUAssignment`** inherits `Tracker` and overrides `data_association` to use IoU for optimal matching.

### Object Detector

The `object_detector.py` class wraps a Faster R-CNN with FPN:

- Uses a **ResNet-50** backbone (no pretrained weights).  
- Provides `detect` method for detecting objects and returning bounding boxes and scores.  
- Parameter `nms_thresh` sets the threshold for merging overlapping detections.

Other scripts (`data_obj_detect.py`, `data_track.py`, `utils.py`) support data handling and utilities but are not central to optimization.

---

### High-Confidence Detections

Filtering low-confidence detections improves tracker performance by reducing false positives. Parameters in Faster R-CNN can be tuned:

- `box_score_thresh`: Minimum classification score for detections.  
- `rpn_nms_thresh`: NMS threshold for RPN proposals.  
- `box_nms_thresh`: NMS threshold for final boxes.  
- `rpn_score_thresh`: Minimum RPN score to consider proposals.

---

### Efficient Data Association

Greedy association can be replaced by the **Hungarian algorithm** to find globally optimal matches:

1. **Distance Matrix**: Rows = new detections, Columns = existing tracks. Cells = similarity (IoU).  
2. **Cost Matrix Extension**: Add rows/columns for missed or false detections with penalties.  
3. **Hungarian Algorithm**: Finds optimal assignment minimizing total cost.  

---

## Implementation

### High-Confidence Detection Tuning

Modified `FRCNN-FPN` parameters:

- `box_score_thresh = 0.81040990`  
- `box_nms_thresh = 0.5`  
- `rpn_nms_thresh = 0.7`  
- `rpn_score_thresh = 0.0`  

Thresholds were experimentally tuned to maximize F1-score.

### Hungarian Tracker

`TrackerIoUAssignment_hungarian` extends `TrackerIoUAssignment`:

- Compute IoU-based distance matrix using `motmetrics.distances.iou_matrix(max_iou=0.5)`.  
- Solve assignment with `scipy.optimize.linear_sum_assignment`.  
- Extend cost matrix to handle false/missing detections (`extend_cost_matrix`).  
- Update existing tracks, add new tracks, and remove missing tracks.

### Video Generation

Function `export_to_mp4` creates videos with tracked detections overlaid.

### Evaluation Functions

- `plot_confusion_matrix(y_pred_proba, y_truth, threshold=0.5)`  
- `plot_precision_recall(y_pred_proba, y_truth)`  
- `calcular_f1_max(y_truth, y_proba)`

### Running the Code

Run the notebook cells sequentially. Model downloads are commented out since local resources were used.

---

## Experimentation

### Object Detector

- Evaluated FRCNN_FPN on **MOT16** test sequences.  
- Optimized classification threshold (`box_score_thresh = 0.81`) improved F1-score to **95.57%**.

![Precision-Recall Curve](Imagenes/curva precision-recall.png)  
*Precision-recall curve for different thresholds.*

![Confusion Matrix](Imagenes/matriz de confusion.png)  
*Confusion matrix for optimal threshold.*

### Hungarian Tracker

- Tracker tested with Hungarian assignment and extended cost matrix.  
- Significant improvement in detection-to-track assignment metrics.

---

## Data Used

**MOT16** dataset:

- Training sequences: MOT16-02, 04, 05, 09  
- Test sequences: MOT16-10, 11, 13  

Realistic videos with multiple object annotations from both fixed and moving cameras.

---

## Results and Analysis

### Object Detector

- Base vs. tuned thresholds show small improvement in metrics.

![Base Detector](Imagenes/object detector base.PNG)  
*Base object detector.*

![Optimized Detector](Imagenes/object detector optim.PNG)  
*Detector with fine-tuned `box_score_thresh`.*

### Tracker

- Base tracker using IoU vs. improved tracker with Hungarian and thresholding shows major improvement in all metrics.  
- Visual comparison confirms better tracking, especially for small or partially occluded objects.

![Base Tracker](Imagenes/evaluación tracker base con IoU.PNG)  
*Base tracker.*

![Tracker with Hungarian](Imagenes/modelo mejorado.PNG)  
*Tracker with Hungarian algorithm and optimized detector.*

- Remaining challenges: occlusions can still cause tracking failures (IDF1, IDP, IDR metrics).

---

## Conclusions

- Hungarian algorithm significantly improves assignment over greedy IoU.  
- Future work: fine-tune all FasterRCNN parameters and implement trajectory prediction with **Kalman filters** to handle occlusion.

---

## Time Log

- Base model analysis and execution: 4 hours  
- Improvement design: 10 hours  
- Implementation and testing: 13 hours  
- Writing report: 8 hours  

---

## References

1. [MOT16 Dataset](https://motchallenge.net/data/MOT16/)  
2. [Computer Vision for Tracking](https://thinkautonomous.medium.com/computer-vision-for-tracking-8220759eee85)  
3. [PyTorch Faster R-CNN](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py)  
4. R.E. Burkard, M. Dell'Amico, S. Martello, *Assignment Problems*, SIAM, 2012.  
5. A. Bewley et al., *Simple online and realtime tracking*, IEEE ICIP, 2016.