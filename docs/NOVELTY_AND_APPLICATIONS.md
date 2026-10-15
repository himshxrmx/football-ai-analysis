# Novelty and practical applications

## What is new compared with typical student projects

* **Camera motion compensation for broadcast video.** Positions are corrected for pans and zooms with optical
  flow and chained homographies, so heatmaps and distances from a moving camera are meaningful. Movement trails
  in the annotated video stay attached to the grass.
* **Honest units.** Values stay in pixels, clearly labelled, until the user calibrates the pitch; only then are
  metres, sprints and comparisons shown.
* **Measured tuning instead of defaults.** The tracker was tuned on real footage (110 to 49 IDs for the same
  detections) and compared with BoT-SORT; detection resolution was compared at 640 and 960 pixels.
* **Explainable coaching.** Every insight names its rule, its evidence and its caveat, and the module refuses to
  make suggestions when the data is too weak.
* **Widest-view reference frame.** The reference view for heatmaps and calibration is chosen automatically as the
  most zoomed-out frame, so most positions fall inside it.
* **Fully local and free.** CPU only, no paid API, video never leaves the computer.

## Practical applications

* **Grassroots and school teams:** objective movement data from a single phone or tripod video.
* **Coaches:** heatmaps and paths to discuss positioning with players after a match.
* **Physical education and sports-science classes:** teaching computer vision with a real application.
* **Scouting preparation:** quick movement overview of a clip before detailed manual review.
* **Research prototypes:** a modular base for testing new trackers, team classifiers or event detectors.
* **Other sports on a flat field:** hockey, rugby or ultimate frisbee with a different landmark set.
