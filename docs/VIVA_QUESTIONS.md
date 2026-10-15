# Viva questions and answers

### Project and design

**1. What does your project do, in one sentence?**
It takes an ordinary football video, detects and tracks the people on the pitch with pretrained AI models,
converts their positions to metres after a pitch calibration and shows movement statistics, heatmaps, player
comparisons and explainable coaching insights in a web dashboard.

**2. Why did you not use a paid API?**
Paid vision APIs charge per image, so analysing every frame of a video becomes expensive; they return generic
labels without tracking or pitch geometry; the video would leave the computer; and in a viva the internal method
could not be explained. YOLO and ByteTrack are free, run locally and are fully explainable.

**3. Why YOLO and not Faster R-CNN or SSD?**
YOLO is a single-stage detector: one pass through the network gives all boxes, so it is much faster than
two-stage detectors like Faster R-CNN, which matters on a CPU. The YOLO family is also actively maintained,
pretrained weights are free, and the same library provides tracking and pose models.

**4. Which classes does your model detect?**
The COCO-pretrained model provides 80 classes; the project uses `person` (class 0) and `sports ball` (class 32).
It cannot separate players, referees and coaches. A football-specific model with classes player, goalkeeper,
referee and ball can be plugged in and is mapped automatically.

**5. What is the difference between detection and tracking?**
Detection finds objects in a single frame and has no memory. Tracking links detections over time so the same
person keeps the same ID, which is needed for any per-player statistic such as distance.

**6. Explain ByteTrack.**
For every frame a Kalman filter predicts each track's position; high-confidence detections are matched to
predictions by IoU with the Hungarian algorithm; unmatched tracks are then matched to low-confidence detections,
which recovers partly hidden players; unmatched high-confidence detections start new tracks; tracks lost for more
than the buffer time are removed.

**7. What is IoU?**
Intersection over union: the area where two boxes overlap divided by the area they cover together. 1 means
identical boxes, 0 means no overlap.

**8. Why are there more track IDs than players?**
When a player is hidden for longer than the track buffer, leaves the view, or two players cross, the tracker
cannot be sure it is the same person and starts a new ID. On the sample clip 49 IDs were created for about 26
people; 37 of them were followed for at least 2 seconds.

**9. Do the IDs tell you who the player is?**
No. IDs are arbitrary numbers. Identifying players would need shirt-number recognition or face recognition,
which is not implemented and is listed as future work.

**10. How did you reduce ID switches?**
I measured it. With the same detections, ByteTrack's default IoU match threshold produced 110 IDs; allowing
matches from IoU 0.1 produced 49. BoT-SORT with camera motion compensation produced 55 and was about 15 times
slower, so ByteTrack with the tuned threshold is used.

### Computer vision and geometry

**11. Why do you need camera motion compensation?**
Broadcast cameras pan and zoom. Without compensation, a player standing still would seem to move across the
image, adding fake distance and smearing heatmaps.

**12. How is camera motion estimated?**
Corner points on the grass are tracked from one frame to the next with Lucas-Kanade optical flow, player boxes
are masked out, and RANSAC fits a homography to the point pairs. Chaining these homographies maps every frame to
one reference view.

**13. What is a homography?**
A 3 x 3 matrix that maps points from one view of a plane to another view of the same plane. It has 8 degrees of
freedom, so 4 point pairs are the minimum to compute it.

**14. Why do you need four points for calibration?**
Each point pair gives two equations (x and y); the homography has 8 unknowns; so 4 pairs are needed. More points
with RANSAC make the result robust to a badly clicked point.

**15. What is RANSAC?**
Random sample consensus. It repeatedly fits the model to a random minimal subset of points, counts how many other
points agree within a tolerance, and keeps the model with the most agreeing points. Wrong matches (outliers) do
not spoil the result.

**16. Why are distances in pixels before calibration, and why is that a problem?**
Without knowing the pitch geometry there is no scale. Also, because of perspective, a player far from the camera
moves fewer pixels than a near player for the same real distance, so pixel values are not comparable between
players. The app labels them clearly and only makes comparisons and suggestions after calibration.

**17. How accurate is your calibration?**
On the sample clip, seven landmarks give a mean reprojection error of 0.22 m, and the projected pitch lines lie on
the real lines in the overlay image.

**18. How do you remove the crowd and the bench?**
A pitch mask: green pixels in HSV space, cleaned with morphology; the convex hull of the largest green region is
the playing area. People whose feet are outside it are dropped.

### Analysis

**19. How is distance calculated?**
Feet positions are mapped to metres, smoothed with a 5-frame moving average and the distances between consecutive
positions are summed. Gaps of more than 3 analysed frames and steps above 12 m/s are not counted.

**20. Why smooth the positions?**
Bounding boxes jitter by a few pixels from frame to frame. Summing that jitter would add distance the player never
ran. The smoothing pads the ends of each track by reflection so straight-line movement is not shortened.

**21. How do you define a sprint?**
At least one second continuously above 7 m/s (25.2 km/h). High-speed running is distance above 5.5 m/s
(19.8 km/h). These thresholds are common in sports science.

**22. How do you separate the two teams?**
Experimental shirt-colour grouping: the median torso colour of each track in CIE Lab space (grass pixels removed),
then k-means clustering. A small third cluster (referees, staff) is left unassigned. On the sample it separated
white and neon-green kits correctly.

### Coaching module and ethics

**23. Is your "AI coach" machine learning?**
It is a rule-based expert system on top of machine-learning measurements. Rules were chosen because every output
can be explained: each insight shows its rule, the measured evidence and a caveat. An optional local language
model can rewrite the facts into a paragraph, but it never creates new facts.

**24. Why does the coach not talk about fitness, passes or possession?**
None of these are measured. Movement speed in a 30-second clip says nothing reliable about fitness, and passes or
possession need validated ball tracking and event detection, which the system does not have. Claiming them would
be fabricating results.

**25. When does the coach refuse to make suggestions?**
When the video is not calibrated, when a player was tracked for less than 5 s or 20% of the clip, when fewer than
3 other stable tracks exist for comparison, and for zone suggestions when there is less than 60 s of tracking.
It then shows data-quality notes instead.

### Engineering

**26. How does the system run without a GPU?**
The nano model has about 2.6 million parameters; PyTorch's CPU build runs it in about 165 ms per 960-pixel frame
on the test laptop. Analysing every second frame and limiting the duration keeps a 30-second clip at about
2 minutes.

**27. What is the trade-off between speed and accuracy?**
Larger model input finds more small players (21.9 instead of 16.4 people per frame at 960 px compared with 640 px)
but is slower; analysing more frames improves tracking but takes longer.

**28. Why does the annotated video play in the browser when OpenCV videos often do not?**
OpenCV's default MP4 codec (mp4v) is not supported by browsers. The project pipes frames into the FFmpeg binary
bundled with imageio-ffmpeg and encodes H.264 with the yuv420p pixel format.

**29. How did you test the project?**
56 automated pytest tests cover validation of bad and corrupted files, video reading and writing, class mapping,
model-loading errors, detection, tracking consistency, camera motion recovery on a shifted image, calibration
accuracy, distance and speed against synthetic motion with known answers, empty data, coaching rule conditions
and PDF generation. The full app was also checked manually in a browser.

**30. What would you improve next?**
Fine-tune YOLO on a labelled football dataset to separate players, referees and the ball; detect pitch keypoints
automatically for per-frame calibration; add re-identification so players keep their IDs; then build validated
event detection such as passes.
