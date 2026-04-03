"""
app.py – Streamlit web application for GPU-accelerated real-time object detection.

Provides an interactive browser UI with three tabs:
  - Image Detection : upload a single image and view annotated results.
  - Video Detection : upload a video clip, process it frame-by-frame, and download the output.
  - Benchmarks      : run CPU-vs-GPU latency comparisons over many inference iterations.

Usage:
    streamlit run app.py
"""
import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from src.detector import ObjectDetector
import time
import tempfile

# Configure the page: descriptive browser-tab title and wide layout for more display space
st.set_page_config(page_title="GPU Object Detection", layout="wide")

st.title("🚀 GPU-Accelerated Object Detection")
st.markdown("**Compare CPU vs GPU performance for real-time object detection**")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Device selection: CPU is always available; CUDA is appended when a compatible GPU is detected
device_options = ['cpu']
if torch.cuda.is_available():
    device_options.append('cuda')
    st.sidebar.success("✅ GPU (CUDA) Available!")
else:
    st.sidebar.warning("⚠️ No GPU detected - CPU only")

selected_device = st.sidebar.selectbox("Select Device", device_options)

# Model selection: 's' is small/fast, 'm' is balanced, 'l' is large/accurate
model_size = st.sidebar.selectbox(
    "Model Size",
    ['yolov5s', 'yolov5m', 'yolov5l'],
    help="Larger models are more accurate but slower"
)

# Initialise detector – @st.cache_resource caches the model across reruns to avoid redundant loading
@st.cache_resource
def load_detector(model, device):
    """Load and cache an ObjectDetector for the specified model and device.

    Args:
        model (str): YOLOv5 variant name, e.g. 'yolov5s', 'yolov5m', or 'yolov5l'.
        device (str): Compute backend – 'cpu' or 'cuda'.

    Returns:
        ObjectDetector: Fully initialised, cached detector instance.
    """
    return ObjectDetector(model_name=model, device=device)

detector = load_detector(model_size, selected_device)

# Main content: three independent tabs covering the core use-cases
tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📊 Benchmarks"])

# TAB 1: Image Detection
with tab1:
    st.header("Upload an Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Open the uploaded bytes as a PIL Image for display and persistence
        image = Image.open(uploaded_file)
        
        # Two-column layout: original image on the left, annotated result on the right
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            # Write the image to a temp file so OpenCV can read it by path inside the detector
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                
                # Run inference and record elapsed wall-clock time
                with st.spinner(f'Running detection on {selected_device.upper()}...'):
                    results, inference_time = detector.detect_image(tmp_file.name)
                
                # Convert the annotated YOLOv5 result frame to a NumPy array for st.image
                result_img = np.array(results.render()[0])
                st.image(result_img, use_column_width=True)
                
                # Surface key performance indicators as Streamlit metric cards
                st.metric("Inference Time", f"{inference_time:.2f} ms")
                st.metric("Device", selected_device.upper())
                
                # Extract bounding-box detections as a DataFrame (xyxy coordinate format)
                detections = results.pandas().xyxy[0]
                st.subheader(f"Detected {len(detections)} objects:")
                st.dataframe(detections[['name', 'confidence']])

# TAB 2: Video Detection
with tab2:
    st.header("Upload a Video")
    st.info("Note: Video processing may take time depending on length and device")
    
    uploaded_video = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        # Persist the uploaded bytes to a temporary file so OpenCV can open it by path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(uploaded_video.read())
            video_path = tmp_video.name
        
        if st.button("Process Video"):
            output_path = "output_video.mp4"
            
            with st.spinner('Processing video... This may take a while'):
                # detect_video returns aggregate timing statistics for the whole clip
                stats = detector.detect_video(video_path, output_path)
            
            st.success("Video processed successfully!")
            
            # Display per-run aggregate statistics across three metric columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg FPS", f"{stats['avg_fps']:.2f}")
            col2.metric("Total Frames", stats['total_frames'])
            col3.metric("Avg Inference", f"{stats['avg_inference_time']:.2f} ms")
            
            # Offer the annotated output video as a browser download
            with open(output_path, 'rb') as f:
                st.download_button(
                    "Download Processed Video",
                    f,
                    file_name="detected_video.mp4"
                )

# TAB 3: Benchmarks
with tab3:
    st.header("Performance Benchmarks")
    st.markdown("Compare CPU vs GPU performance")
    
    if st.button("Run Benchmark (100 iterations)"):
        from src.benchmark import PerformanceBenchmark
        
        # Use a static sample image as the repeated input for every benchmark iteration
        st.info("Running benchmark... this will take a few minutes")
        
        # Initialise the benchmark runner with the bundled test image
        benchmark = PerformanceBenchmark("data/sample_images/test.jpg")
        
        with st.spinner("Benchmarking..."):
            results = benchmark.run_benchmark(num_iterations=100)
        
        # Show CPU vs GPU side-by-side comparison only when both devices were benchmarked
        if 'cuda' in results and 'cpu' in results:
            # Compute overall mean speedup to headline the results section
            speedup = results['cpu']['mean'] / results['cuda']['mean']
            st.success(f"🚀 GPU is {speedup:.2f}x faster than CPU!")
            
            # Display CPU and GPU latency metrics in two equal-width columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("CPU Performance")
                st.metric("Mean Inference", f"{results['cpu']['mean']:.2f} ms")
                st.metric("Median", f"{results['cpu']['median']:.2f} ms")
                st.metric("95th Percentile", f"{results['cpu']['p95']:.2f} ms")
            
            with col2:
                st.subheader("GPU Performance")
                st.metric("Mean Inference", f"{results['cuda']['mean']:.2f} ms")
                st.metric("Median", f"{results['cuda']['median']:.2f} ms")
                st.metric("95th Percentile", f"{results['cuda']['p95']:.2f} ms")
        else:
            # Fallback: display CPU-only results when no CUDA device is present
            st.warning("GPU not available for comparison")
            st.metric("CPU Mean Inference", f"{results['cpu']['mean']:.2f} ms")