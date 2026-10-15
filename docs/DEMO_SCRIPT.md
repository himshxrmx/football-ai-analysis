# Demonstration script (about 5 minutes)

**Before the demo:** start the app (`run_app.bat`), open http://localhost:8501 and make sure the sample analysis
already exists (processing takes about 2 minutes, so run it beforehand and only show the detection check live).

### 1. Analyse a video (1 minute)
* "I choose the sample clip: a 30-second Bundesliga clip, 1080p, 25 frames per second. The system has checked the
  file type, size, duration and that frames can be decoded."
* Show the settings: "Model size, confidence threshold, processing resolution and frame step control the trade-off
  between speed and accuracy."
* Click **Run detection on this frame**: "YOLO26 finds about 23 people and the ball in under 200 ms on this
  laptop's CPU. Each box shows its confidence."
* Point at the estimate: "A full run takes about 2 minutes; I have already run it."

### 2. Overview (30 seconds)
* "22.8 people were tracked per frame on average; 49 track IDs, 37 of them stable. More IDs than players because
  IDs restart after occlusions. Camera motion was estimated for every frame."

### 3. Player tracking (45 seconds)
* Play the annotated video: "IDs, confidence and movement trails. The trails stay on the grass while the camera
  pans, because they are projected through the camera homography."
* Move the frame slider and show the detection table.

### 4. Pitch calibration (45 seconds)
* "The pitch is flat, so a homography converts pixels to metres. I marked seven landmarks."
* Show the overlay: "The yellow projected lines sit on the real pitch lines; the mean error is 0.22 m."

Landmarks used for the sample clip (reference frame = video frame 0):

| Landmark | x (px) | y (px) |
|---|---|---|
| Halfway line, far touchline | 887 | 366 |
| Halfway line, near touchline | 927 | 760 |
| Centre circle, far point | 895 | 440 |
| Centre circle, near point | 903 | 529 |
| Right penalty area, far front corner | 1454 | 397 |
| Right penalty area, near front corner | 1808 | 587 |
| Left penalty area, far front corner | 319 | 410 |

### 5. Heatmaps, statistics, comparison (1 minute)
* Heatmap: "Most activity was around the centre circle and the right half."
* Movement paths of three IDs.
* Statistics: "Average speeds of 2 to 3.3 m/s and peaks up to 7.2 m/s, realistic values for professional players."
* Comparison: pick two IDs and show the small multiples.

### 6. AI coach (45 seconds)
* Whole clip: measured facts.
* Single player: pick a low-speed ID. "Each card says which rule fired and with which numbers, and the caveat. It
  never talks about fitness, passes or possession because those are not measured."

### 7. Reports (15 seconds)
* Build the PDF and show the CSV downloads.

**Closing line:** "Everything runs locally on a CPU with free, open-source models, and every number on screen
comes from the video."
