# VISIONPORT Object Detection — Captured Results

## Run metadata

- Notebook: `Skill_1.ipynb`
- Notebook cell execution count: `2`
- Model: Ultralytics YOLOv8 Nano (`yolov8n.pt`)
- Prediction confidence threshold: `0.25`
- Inference image size: `640`
- Input image dimensions reported by OpenCV: `(424, 724, 3)`
- Total detections: **21**

## Captured detections

| # | Class | Confidence | Bounding box (x1, y1, x2, y2) |
|---:|---|---:|---|
| 1 | car | 92.46% | (288, 163, 435, 223) |
| 2 | car | 89.14% | (193, 154, 350, 220) |
| 3 | person | 77.06% | (657, 324, 705, 393) |
| 4 | car | 76.30% | (106, 213, 207, 288) |
| 5 | person | 74.93% | (378, 329, 425, 406) |
| 6 | car | 74.73% | (179, 314, 321, 419) |
| 7 | person | 71.76% | (509, 271, 553, 348) |
| 8 | person | 71.16% | (61, 272, 97, 327) |
| 9 | car | 65.94% | (76, 191, 154, 252) |
| 10 | motorcycle | 65.26% | (659, 369, 706, 423) |
| 11 | motorcycle | 63.14% | (383, 370, 423, 422) |
| 12 | motorcycle | 60.45% | (454, 175, 499, 216) |
| 13 | motorcycle | 59.97% | (513, 307, 556, 367) |
| 14 | car | 59.91% | (144, 172, 209, 211) |
| 15 | motorcycle | 55.93% | (66, 302, 97, 347) |
| 16 | person | 54.42% | (404, 210, 442, 269) |
| 17 | truck | 48.63% | (16, 165, 64, 227) |
| 18 | bus | 48.31% | (16, 166, 64, 227) |
| 19 | motorcycle | 47.50% | (405, 235, 441, 283) |
| 20 | truck | 36.15% | (113, 154, 156, 191) |
| 21 | truck | 35.13% | (72, 162, 116, 204) |

## Captured system report

The notebook reported success for image acquisition, preprocessing, YOLOv8 model loading, object detection, bounding-box generation, and output visualization. It reported 21 objects detected and completed the VISIONPORT pipeline.

## Notebook outputs

The uploaded notebook also contained:
- A Google Colab upload widget output.
- A Matplotlib figure showing the uploaded image beside the YOLOv8 detection visualization.
- A Colab download action for `VisionPort_Detection_Output.jpg`.

The upload/download widgets are session-specific Colab UI outputs. The notebook file in this repository retains the executable code; rerun it in Colab to activate those widgets and regenerate the image.

## Reproducibility notes

The captured detections above are the recorded output from the uploaded notebook execution. They are not a guarantee that a later run will produce identical confidences: model/library versions, input image bytes, and runtime conditions can affect results.
