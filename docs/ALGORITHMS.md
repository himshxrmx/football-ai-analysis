# Algorithms used, explained simply

## 1. YOLO object detection (YOLO26)

**What it does.** Looks at one image and returns a box, a class (person, sports ball) and a confidence score for
every object it recognises.

**How it works.** YOLO ("You Only Look Once") is a convolutional neural network. The image is resized (for
example to 960 pixels) and passed once through the network. A *backbone* extracts visual features (edges,
textures, body shapes), a *neck* combines features from several scales so both large and small objects are
seen, and a *head* predicts, for many positions in the image, the box coordinates, the class probabilities and
the confidence. Overlapping predictions of the same object are reduced to one box. YOLO26 is the current
Ultralytics generation; its nano variant (2.6 million parameters) is small enough for a CPU. The weights used here
were learned on the COCO dataset (about 118,000 training images, 80 classes).

**Why it was chosen.** Free, pretrained, runs on a CPU, one line to load, well documented.

**Confidence threshold.** Boxes below the threshold are ignored. Lower values find more distant players but add
false detections. In this project boxes down to 0.10 are passed to the tracker (see ByteTrack), but only boxes
above the user threshold can start a track.

## 2. ByteTrack multi-object tracking

**What it does.** Gives the same ID to the same person in consecutive frames.

**Steps per frame.**

1. **Predict.** A Kalman filter predicts where every existing track should be now, using its previous position
   and velocity.
2. **First association.** High-confidence boxes are matched to the predicted boxes. The cost is 1 - IoU
   (intersection over union, the overlap of two boxes). The Hungarian algorithm finds the cheapest one-to-one
   matching.
3. **Second association.** Tracks that are still unmatched are matched to the *low*-confidence boxes. This is
   ByteTrack's key idea: a partly hidden player often gets a low score, and using those boxes keeps the track
   alive instead of losing it.
4. **New tracks** start from unmatched high-confidence boxes.
5. **Lost tracks** are kept for a buffer time (2 seconds here) in case the player reappears, then deleted.

**Tuning in this project.** Players in a wide shot are small and move a large fraction of their box width between
analysed frames, so overlaps are small. Allowing matches from IoU 0.1 (threshold 0.9) reduced the IDs on the
sample clip from 110 to 49.

## 3. Kalman filter

A Kalman filter estimates a hidden state (position and velocity of a box) from noisy measurements. Each step has
a *predict* phase (move the state forward with constant velocity, increase the uncertainty) and an *update* phase
(combine the prediction with the new detection, weighted by their uncertainties). It smooths jittery boxes and
lets the tracker guess where a briefly hidden player is.

## 4. Hungarian algorithm

Solves the assignment problem: given a cost matrix between N tracks and M detections, find the pairing with the
lowest total cost. Used inside ByteTrack (through the `lap` package).

## 5. Pitch filter (colour segmentation)

The frame is converted to HSV colour space and pixels with a green hue are marked. Small holes are closed with
morphological operations, the largest green region is taken as the pitch and its convex hull is filled. A person
whose feet (bottom centre of the box) are outside that hull is in the stands or on the bench and is ignored.

## 6. Camera motion compensation (optical flow and homography)

1. **Shi-Tomasi corners** ("good features to track") are found on a half-size grey image, only on the grass and
   its lines, with player boxes masked out.
2. **Pyramidal Lucas-Kanade optical flow** finds where each corner moved in the next frame, assuming the
   brightness of a small patch stays the same.
3. **RANSAC** (random sample consensus) repeatedly fits a model to random subsets of point pairs and keeps the
   model most points agree with, so wrong matches are ignored.
4. The model is a **homography**, a 3 x 3 matrix that maps one view of a plane to another view of the same
   plane. Because the pitch is flat, a camera pan or zoom is described exactly by a homography.
5. Frame-to-frame homographies are multiplied together, so every frame can be mapped to the first frame and then
   to the widest frame of the clip (the reference view).

## 7. Pitch calibration (homography to metres)

The user marks at least four pitch landmarks (for example the centre-circle points and penalty-area corners) in
the reference frame. Their real positions on a 105 x 68 m pitch are known. OpenCV's `findHomography` solves for
the 3 x 3 matrix H that maps pixels to metres (with RANSAC when more than four points are given). The mean
distance between known and mapped landmarks is reported as the reprojection error. Projecting the pitch lines
back into the image is a visual check.

## 8. Movement measures

* **Position**: bottom centre of the box (where the feet touch the ground), mapped to metres.
* **Smoothing**: a centred 5-frame moving average. The ends of each track are padded by point reflection, so a
  player moving in a straight line keeps the correct distance.
* **Distance**: sum of the distances between consecutive smoothed positions. Steps across gaps of more than
  3 analysed frames and steps faster than 12 m/s (impossible for a human, so a tracking error) are not counted.
* **Speed**: distance divided by time. **Peak speed** is the 95th percentile of step speeds, which ignores
  single-frame spikes.
* **High-speed running**: distance covered above 5.5 m/s (19.8 km/h). **Sprint**: at least 1 s above 7 m/s
  (25.2 km/h). These are common thresholds in sports science.
* **Area covered**: area of the convex hull around all positions of a player.
* **Zones**: share of positions in the left, middle and right third (along the length) and the far, central and
  near channel (across the width).

## 9. Heatmap

A 2D histogram counts positions in 1 x 1 m cells. A Gaussian blur spreads each count over neighbouring cells, and
the result is scaled to 0 to 1 and drawn with a single-hue colour scale (light blue = little time, dark blue = a
lot of time).

## 10. Shirt-colour grouping (k-means clustering)

For each tracked box the upper-torso pixels are converted to CIE Lab colour space (closer to human colour
perception than RGB), pixels close to the measured grass colour are removed, and the median colour is stored.
Every track gets its median shirt colour. k-means clustering (k = 3, then k = 2 if needed) splits the tracks into
colour groups; a small third cluster (referees, staff) and tracks far from both team centres stay unassigned.

## 11. Rule-based coaching (expert system)

A small expert system: each rule has a condition on measured values (for example "average speed below 75% of the
group median, with at least 5 s of tracking and a calibration"), an output (observation or suggestion with a
drill), the evidence values and a caveat. Because the rules are explicit, every output can be explained, which a
black-box model could not do.

## 12. Pose estimation (optional, YOLO26 pose)

A YOLO model trained on COCO keypoints predicts 17 body points (nose, eyes, ears, shoulders, elbows, wrists, hips,
knees, ankles) for each person. The player is cropped and enlarged first. Joint angles are computed with the dot
product of the two limb vectors and are 2D image angles.
