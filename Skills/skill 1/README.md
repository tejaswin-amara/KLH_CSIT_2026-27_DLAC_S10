# Skill 1 — VISIONPORT Object Detection

This folder contains the Skill 1 object-detection implementation and its captured execution results.

## Files

- `visionport_object_detection.ipynb` — runnable Jupyter/Google Colab notebook.
- `visionport_object_detection.py` — Python source extracted from the uploaded notebook cell.
- `detection_results.md` — captured 21 detections, confidence scores, bounding boxes, and run notes.

## Run

The implementation uses Google Colab upload/download helpers. Open the notebook in Google Colab and run its code cell. It installs Ultralytics, loads `yolov8n.pt`, accepts an uploaded photograph, runs prediction, displays the original and annotated images, and downloads `VisionPort_Detection_Output.jpg`.

## Captured run

The recorded run detected 21 objects. See [detection_results.md](detection_results.md) for the complete list.

The notebook's original output included session-bound Colab upload/download widgets and a Matplotlib visualization. The committed notebook contains the executable code without stale session widget output; rerun it to regenerate those outputs.
