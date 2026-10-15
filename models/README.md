# Model weights

Weights are not stored in the repository.

* `yolo26n.pt`, `yolo26s.pt`, `yolo11n.pt` and `yolo26n-pose.pt` are official Ultralytics COCO weights. They are
  downloaded automatically into this folder from the Ultralytics GitHub release page the first time they are used
  (about 5 to 20 MB each, internet needed once).
* `football.pt` (optional): a model fine-tuned on football classes such as `player`, `goalkeeper`, `referee` and
  `ball`. Place it here, or set the environment variable `FOOTBALL_MODEL_PATH`, and it appears in the model list on
  the Analyse page. See `notebooks/train_football_yolo_colab.ipynb` for how to train one on Google Colab.
