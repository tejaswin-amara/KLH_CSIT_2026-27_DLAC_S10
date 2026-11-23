# VISIONPORT Notebooks — Markdown Summary

> **Note:** In the provided notebook JSON payloads, cells `jAMQsZBfiR3j` (`execution_count: 2`) and `A_5jmyX0jc7v` (`execution_count: 1`) did not include a `"source"` code array—only `"outputs"` (`stdout` and `display_data` containing HTML upload helpers and PNG figures). Cell `8p-GlS_xAIvd` (`execution_count: 3`) contained the Python benchmark script (`source`), `stdout`, and a stacked bar chart PNG.

---

## 1. VISIONPORT Object Detection System (`id: jAMQsZBfiR3j`, Execution 2)

### Notebook & Cell Metadata

- **Format:** `nbformat` 4.0, Python 3 kernel.
- **Cell ID:** `jAMQsZBfiR3j` (`execution_count: 2`).
- **Source Code:** Not present in the provided cell JSON.

### Captured `stdout`

```text
Creating new Ultralytics Settings v0.0.8 file ✅
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/usage/settings
============================================================
      VISIONPORT OBJECT DETECTION SYSTEM
============================================================

Upload a real photograph...
Saving images (6).jpeg to images (6) (1).jpeg

Loading YOLOv8 pretrained model...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt to 'yolov8n.pt': 100% ━━━━━━━━━━━━ 6.2MB 72.4MB/s 0.1s
Model loaded successfully!

Image Acquisition Completed
Original Image Size: (424, 724, 3)

Performing Object Detection...

OBJECT DETECTION RESULTS
--------------------------------------------------
1. Object: car
   Confidence: 92.46%
   Bounding Box: (288, 163, 435, 223)
2. Object: car
   Confidence: 89.14%
   Bounding Box: (193, 154, 350, 220)
3. Object: person
   Confidence: 77.06%
   Bounding Box: (657, 324, 705, 393)
4. Object: car
   Confidence: 76.30%
   Bounding Box: (106, 213, 207, 288)
5. Object: person
   Confidence: 74.93%
   Bounding Box: (378, 329, 425, 406)
6. Object: car
   Confidence: 74.73%
   Bounding Box: (179, 314, 321, 419)
7. Object: person
   Confidence: 71.76%
   Bounding Box: (509, 271, 553, 348)
8. Object: person
   Confidence: 71.16%
   Bounding Box: (61, 272, 97, 327)
9. Object: car
   Confidence: 65.94%
   Bounding Box: (76, 191, 154, 252)
10. Object: motorcycle
    Confidence: 65.26%
    Bounding Box: (659, 369, 706, 423)
11. Object: motorcycle
    Confidence: 63.14%
    Bounding Box: (383, 370, 423, 422)
12. Object: motorcycle
    Confidence: 60.45%
    Bounding Box: (454, 175, 499, 216)
13. Object: motorcycle
    Confidence: 59.97%
    Bounding Box: (513, 307, 556, 367)
14. Object: car
    Confidence: 59.91%
    Bounding Box: (144, 172, 209, 211)
15. Object: motorcycle
    Confidence: 55.93%
    Bounding Box: (66, 302, 97, 347)
16. Object: person
    Confidence: 54.42%
    Bounding Box: (404, 210, 442, 269)
17. Object: truck
    Confidence: 48.63%
    Bounding Box: (16, 165, 64, 227)
18. Object: bus
    Confidence: 48.31%
    Bounding Box: (16, 166, 64, 227)
19. Object: motorcycle
    Confidence: 47.50%
    Bounding Box: (405, 235, 441, 283)
20. Object: truck
    Confidence: 36.15%
    Bounding Box: (113, 154, 156, 191)
21. Object: truck
    Confidence: 35.13%
    Bounding Box: (72, 162, 116, 204)
```

### Display Outputs

- **Colab File Upload Widget (`display_data`):** Renders `<input type="file" id="files-9a5cf9bd-4fae-4589-9249-31b0ef8bf839" name="files[]" multiple disabled>` with a status `<output>` and the `google.colab._files` chunked upload helper script (`MAX_PAYLOAD_SIZE = 100 * 1024`).
- **Detection Figure (`display_data`):** Matplotlib figure (`1600x800` with 2 axes) embedded as a base64 PNG (`image/png`).

---

## 2. VISIONPORT — End-to-End System Architecture (`id: A_5jmyX0jc7v`, Execution 1)

### Notebook & Cell Metadata

- **Format:** `nbformat` 4.0, Python 3 kernel.
- **Cell ID:** `A_5jmyX0jc7v` (`execution_count: 1`).
- **Source Code:** Not present in the provided cell JSON.

### Captured `stdout`

```text
Creating new Ultralytics Settings v0.0.8 file ✅
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/usage/settings
=================================================================
      VISIONPORT - END-TO-END SYSTEM ARCHITECTURE
=================================================================

SYSTEM CONFIGURATION
---------------------------------------------
PyTorch Version: 2.14.1+cpu
GPU Available: False
Hardware: CPU
TensorRT: Not installed
Triton Client Installed: False
Triton Server Running: False

UPLOAD YOUR INPUT IMAGE
Saving images (6).jpeg to images (6).jpeg

Loading YOLOv8 Model...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt to 'yolov8n.pt': 100% ━━━━━━━━━━━━ 6.2MB 28.0MB/s 0.2s
YOLOv8 Model Loaded Successfully!

FASTAPI SERVER STARTED SUCCESSFULLY!
API Address: http://127.0.0.1:8000
Health Endpoint: /health
Prediction Endpoint: /predict

Sending Image Through HTTP POST Request...

API RESPONSE RECEIVED!
Status: success
Total Objects: 21
Inference Time: 7250.29 ms
```

### Display Outputs

- **Colab File Upload Widget (`display_data`):** Renders `<input type="file" id="files-6964f63e-4ad8-4ce4-ade7-b468eb2674c3" name="files[]" multiple disabled>` with `<output>` and the `google.colab._files` upload script (`MAX_PAYLOAD_SIZE = 100 * 1024`).
- **Architecture / Inference Figure (`display_data`):** Matplotlib figure (`1600x700` with 2 axes) embedded as a base64 PNG (`image/png`).

---

## 3. VisionPort Lifecycle Component Breakdown Benchmark (`id: 8p-GlS_xAIvd`, Execution 3)

### Notebook & Cell Metadata

- **Format:** `nbformat` 4.0, Python 3 kernel.
- **Cell ID:** `8p-GlS_xAIvd` (`execution_count: 3`).

### Source Code (`run_advanced_benchmark`)

```python
import time
import numpy as np
import matplotlib.pyplot as plt

def run_advanced_benchmark():
    print("Running VisionPort Lifecycle Component Breakdown Benchmark...")

    # Concurrency levels tested
    concurrency_levels = [1, 4, 8, 16]

    # Simulated realistic latencies (ms) for each lifecycle stage across concurrency
    # [API Ingress, Preprocessing, GPU Inference (TensorRT), Postprocessing]
    ingress_latencies = [1.2, 1.5, 2.1, 3.4]
    preprocessing_latencies = [4.5, 4.8, 5.2, 6.0]
    gpu_inference_latencies = [8.3, 9.1, 11.5, 15.2]
    postprocessing_latencies = [1.4, 1.6, 1.9, 2.3]

    x = np.arange(len(concurrency_levels))
    width = 0.55

    # --- Generate Stacked Bar Chart for Lifecycle Breakdown ---
    fig, ax = plt.subplots(figsize=(10, 6))

    p1 = ax.bar(x, ingress_latencies, width, label='API Ingress (FastAPI)', color='#3b82f6')
    p2 = ax.bar(x, preprocessing_latencies, width, bottom=ingress_latencies, label='Preprocessing (OpenCV/NumPy)', color='#10b981')

    bottom_gpu = np.array(ingress_latencies) + np.array(preprocessing_latencies)
    p3 = ax.bar(x, gpu_inference_latencies, width, bottom=bottom_gpu, label='GPU Inference (TensorRT)', color='#f59e0b')

    bottom_post = bottom_gpu + np.array(gpu_inference_latencies)
    p4 = ax.bar(x, postprocessing_latencies, width, bottom=bottom_post, label='Postprocessing & NMS', color='#ef4444')

    ax.set_ylabel('Latency (Milliseconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Concurrency Level', fontsize=12, fontweight='bold')
    ax.set_title('VisionPort CO1 Lifecycle Stage Breakdown Across Concurrency', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concurrency_levels])
    ax.legend(loc='upper left', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add total latency text annotations on top of each bar
    for i in range(len(concurrency_levels)):
        total = ingress_latencies[i] + preprocessing_latencies[i] + gpu_inference_latencies[i] + postprocessing_latencies[i]
        ax.text(x[i], total + 0.5, f"{total:.1f} ms", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    output_filename = "visionport_lifecycle_breakdown.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Success! New component breakdown graph saved as '{output_filename}'.")
    plt.show()

if __name__ == "__main__":
    run_advanced_benchmark()
```

### Captured `stdout`

```text
Running VisionPort Lifecycle Component Breakdown Benchmark...
Success! New component breakdown graph saved as 'visionport_lifecycle_breakdown.png'.
```

### Display Outputs

- **Stacked Bar Chart (`display_data`):** Matplotlib figure (`1000x600` with 1 axis) embedded as a base64 PNG (`image/png`), showing total latencies across concurrency levels `[1, 4, 8, 16]`: `15.4 ms`, `17.0 ms`, `20.7 ms`, and `26.9 ms`.

> **Benchmark qualification:** The benchmark source explicitly labels the stage latencies as simulated. Treat the resulting values as illustrative measurements, not as empirical GPU/TensorRT performance results. The architecture notebook's captured environment also reports CPU-only execution, no TensorRT installation, and no running Triton server.
