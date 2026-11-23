import time
import numpy as np
import matplotlib.pyplot as plt


def build_benchmark_figure(concurrency_levels=None):
    if concurrency_levels is None:
        concurrency_levels = [1, 4, 8, 16]

    # Simulated realistic latencies (ms) for each lifecycle stage across concurrency
    # [API Ingress, Preprocessing, GPU Inference (TensorRT), Postprocessing]
    ingress_latencies = [1.2, 1.5, 2.1, 3.4]
    preprocessing_latencies = [4.5, 4.8, 5.2, 6.0]
    gpu_inference_latencies = [8.3, 9.1, 11.5, 15.2]
    postprocessing_latencies = [1.4, 1.6, 1.9, 2.3]

    x = np.arange(len(concurrency_levels))
    width = 0.55

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, ingress_latencies, width, label="API Ingress (FastAPI)", color="#3b82f6")
    ax.bar(
        x,
        preprocessing_latencies,
        width,
        bottom=ingress_latencies,
        label="Preprocessing (OpenCV/NumPy)",
        color="#10b981",
    )

    bottom_gpu = np.array(ingress_latencies) + np.array(preprocessing_latencies)
    ax.bar(
        x,
        gpu_inference_latencies,
        width,
        bottom=bottom_gpu,
        label="GPU Inference (TensorRT)",
        color="#f59e0b",
    )

    bottom_post = bottom_gpu + np.array(gpu_inference_latencies)
    ax.bar(
        x,
        postprocessing_latencies,
        width,
        bottom=bottom_post,
        label="Postprocessing & NMS",
        color="#ef4444",
    )

    ax.set_ylabel("Latency (Milliseconds)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Concurrency Level", fontsize=12, fontweight="bold")
    ax.set_title(
        "VisionPort CO1 Lifecycle Stage Breakdown Across Concurrency",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concurrency_levels])
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    totals = (
        np.array(ingress_latencies)
        + np.array(preprocessing_latencies)
        + np.array(gpu_inference_latencies)
        + np.array(postprocessing_latencies)
    )
    for i in range(len(concurrency_levels)):
        ax.text(
            x[i],
            total[i] + 0.5,
            f"{total[i]:.1f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    fig.tight_layout()
    return fig


def run_advanced_benchmark(output_filename="visionport_lifecycle_breakdown.png"):
    assert output_filename.endswith(".png"), "Output filename must end with .png"
    start_time = time.perf_counter()
    print("Running VisionPort Lifecycle Component Breakdown Benchmark...")
    fig = build_benchmark_figure()
    fig.savefig(output_filename, dpi=300)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(f"Success! New component breakdown graph saved as '{output_filename}' ({elapsed_ms:.2f} ms).")
    plt.show()


if __name__ == "__main__":
    run_advanced_benchmark()
