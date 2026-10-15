# Presentation outline (about 12 slides, 10 minutes)

1. **Title.** Football Video Analysis System using YOLO and ByteTrack. Names, guide, department, year.

2. **Problem.** Teams have match video but no affordable movement data. Manual analysis is slow; GPS vests and
   stadium tracking systems are expensive.

3. **Objectives.** Validate video; detect and track players; heatmaps and pitch view; metres only with
   calibration; dashboard and comparison; explainable coaching; CSV and PDF export; CPU only, no paid API.

4. **Existing systems and their limits.** Table: manual analysis, wearables, optical tracking products, cloud
   vision APIs, and what each lacks.

5. **Architecture.** The block diagram from the README: interface, processing modules, local file storage.

6. **Pipeline.** Video, YOLO26 detection, pitch filter, ByteTrack, camera motion compensation, calibration,
   analysis, dashboard. One sentence per step.

7. **Key algorithms.** YOLO single-pass detection; ByteTrack two-stage association; optical flow plus RANSAC
   homography; calibration homography. Use simple diagrams.

8. **Results.** Sample clip numbers: 22.8 people tracked per frame, 3.1 frames per second on a laptop CPU,
   calibration error 0.22 m, tracker tuning 110 to 49 IDs. Screenshot of the heatmap and paths.

9. **Dashboard screenshots.** Overview, tracking frame with IDs, statistics table, comparison, AI coach card with
   "Why this was generated".

10. **Honesty by design.** Pixels vs metres; IDs are not player identities; what is not measured (passes,
    possession, fitness); suggestions only with enough data.

11. **Testing.** 56 automated tests and what they cover; manual browser test.

12. **Limitations and future work.** Occlusion, COCO classes, ball detection, manual calibration; fine-tuned
    football model, automatic pitch keypoints, re-identification, event detection.

13. **Live demo** (see DEMO_SCRIPT.md), then questions.
