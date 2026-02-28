import streamlit as st
import torch
from PIL import Image
import cv2
import numpy as np
from src.detector import ObjectDetector
import time
import tempfile

st.set_page_config(page_title="GPU Object Detection", layout="wide")

st.title("🚀 GPU-Accelerated Object Detection")
st.markdown("**Compare CPU vs GPU performance for real-time object detection**")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# Device selection
device_options = ['cpu']
if torch.cuda.is_available():
    device_options.append('cuda')
    st.sidebar.success("✅ GPU (CUDA) Available!")
else:
    st.sidebar.warning("⚠️ No GPU detected - CPU only")

selected_device = st.sidebar.selectbox("Select Device", device_options)

# Model selection
model_size = st.sidebar.selectbox(
    "Model Size",
    ['yolov5s', 'yolov5m', 'yolov5l'],
    help="Larger models are more accurate but slower"
)

# Initialize detector
@st.cache_resource
def load_detector(model, device):
    return ObjectDetector(model_name=model, device=device)

detector = load_detector(model_size, selected_device)

# Main content
tab1, tab2, tab3 = st.tabs(["📷 Image Detection", "🎥 Video Detection", "📊 Benchmarks"])

# TAB 1: Image Detection
with tab1:
    st.header("Upload an Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                
                # Run detection
                with st.spinner(f'Running detection on {selected_device.upper()}...'):
                    results, inference_time = detector.detect_image(tmp_file.name)
                
                # Display results
                result_img = np.array(results.render()[0])
                st.image(result_img, use_column_width=True)
                
                # Performance metrics
                st.metric("Inference Time", f"{inference_time:.2f} ms")
                st.metric("Device", selected_device.upper())
                
                # Detection details
                detections = results.pandas().xyxy[0]
                st.subheader(f"Detected {len(detections)} objects:")
                st.dataframe(detections[['name', 'confidence']])

# TAB 2: Video Detection
with tab2:
    st.header("Upload a Video")
    st.info("Note: Video processing may take time depending on length and device")
    
    uploaded_video = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video is not None:
        # Save uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(uploaded_video.read())
            video_path = tmp_video.name
        
        if st.button("Process Video"):
            output_path = "output_video.mp4"
            
            with st.spinner('Processing video... This may take a while'):
                stats = detector.detect_video(video_path, output_path)
            
            st.success("Video processed successfully!")
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg FPS", f"{stats['avg_fps']:.2f}")
            col2.metric("Total Frames", stats['total_frames'])
            col3.metric("Avg Inference", f"{stats['avg_inference_time']:.2f} ms")
            
            # Download button for processed video
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
        
        # Use a sample image for benchmarking
        st.info("Running benchmark... this will take a few minutes")
        
        # You'll need to provide a sample image path
        benchmark = PerformanceBenchmark("data/sample_images/test.jpg")
        
        with st.spinner("Benchmarking..."):
            results = benchmark.run_benchmark(num_iterations=100)
        
        # Display results
        if 'cuda' in results and 'cpu' in results:
            speedup = results['cpu']['mean'] / results['cuda']['mean']
            st.success(f"🚀 GPU is {speedup:.2f}x faster than CPU!")
            
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
            st.warning("GPU not available for comparison")
            st.metric("CPU Mean Inference", f"{results['cpu']['mean']:.2f} ms")