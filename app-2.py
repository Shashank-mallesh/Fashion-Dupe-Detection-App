"""
Fashion Dupe Detection - Streamlit Application
Simplified version that works without external dependencies
"""
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(
    page_title="Fashion Dupe Detection",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def mock_predict_class(image, class_names):
    """Mock prediction function that works without a real model"""
    # Simple mock based on image properties
    if image:
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Mock logic based on image characteristics
        if len(img_array.shape) == 3:
            # Analyze color distribution
            avg_color = np.mean(img_array, axis=(0, 1))
            
            # Simple mock classification
            if avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:  # More red
                predicted_idx = 2  # women
            elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:  # More green
                predicted_idx = 1  # men
            else:
                predicted_idx = 0  # footwear
        else:
            predicted_idx = 0  # footwear
        
        # Generate mock probabilities
        probs = np.random.dirichlet(np.ones(3) * 10)
        probs[predicted_idx] += 0.3  # Boost the predicted class
        probs = probs / probs.sum()
        
        confidence = probs[predicted_idx] * 100
        return class_names[predicted_idx], confidence, probs
    
    return "unknown", 0.0, np.array([0.33, 0.33, 0.34])

def mock_compute_similarity(image1, image2):
    """Mock similarity computation"""
    if image1 and image2:
        # Convert images to numpy arrays
        img1_array = np.array(image1)
        img2_array = np.array(image2)
        
        # Simple mock similarity based on image properties
        if img1_array.shape == img2_array.shape:
            # Calculate color histogram similarity
            hist1 = np.histogram(img1_array.flatten(), bins=50)[0]
            hist2 = np.histogram(img2_array.flatten(), bins=50)[0]
            
            # Normalize histograms
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()
            
            # Calculate similarity (1 - histogram distance)
            similarity = 1.0 - 0.5 * np.sum(np.abs(hist1 - hist2))
            similarity = max(0.3, min(0.95, similarity))  # Keep in reasonable range
        else:
            similarity = 0.5 + np.random.random() * 0.3  # Random similarity
        
        return similarity * 100
    
    return 50.0

def is_dupe(similarity, threshold):
    """Determine if images are dupes based on similarity threshold"""
    return similarity >= threshold

def main():
    # Header
    st.title("👗 Fashion Dupe Detection System")
    st.markdown("### Identify fashion items and detect duplicate/similar products")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["Single Image Classification", "Dupe Detection (Compare 2 Images)"]
        )
        
        # Threshold for dupe detection
        if mode == "Dupe Detection (Compare 2 Images)":
            threshold = st.slider(
                "Similarity Threshold (%)",
                min_value=50,
                max_value=100,
                value=75,
                step=5,
                help="Images above this threshold are considered dupes"
            )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This app demonstrates fashion item classification and duplicate detection. "
            "Currently running in demo mode with mock predictions."
        )
    
    # Class names
    class_names = ['footwear', 'men', 'women']
    
    # Main content
    if mode == "Single Image Classification":
        st.header("📸 Single Image Classification")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose a fashion item image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload an image of a fashion item (clothing or footwear)"
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
        
        with col2:
            if uploaded_file is not None:
                st.subheader("Prediction Results")
                
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    predicted_class, confidence, all_probs = mock_predict_class(
                        image, class_names
                    )
                
                # Display results
                st.markdown(f"""
                <div class="result-box">
                    <h3>Category: {predicted_class.upper()}</h3>
                    <h4>Confidence: {confidence:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability chart
                st.subheader("Class Probabilities")
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                bars = ax.barh(class_names, all_probs * 100, color=colors)
                ax.set_xlabel('Probability (%)')
                ax.set_xlim(0, 100)
                ax.set_title('Prediction Confidence by Category')
                
                # Add value labels
                for bar, prob in zip(bars, all_probs):
                    width = bar.get_width()
                    ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                           f'{prob*100:.1f}%', ha='left', va='center')
                
                st.pyplot(fig)
            else:
                st.info("Please upload an image to see classification results")
    
    else:  # Dupe Detection Mode
        st.header("🔍 Dupe Detection - Compare Two Images")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Item")
            image1_file = st.file_uploader(
                "Upload first image",
                type=['png', 'jpg', 'jpeg'],
                key="img1"
            )
            if image1_file is not None:
                try:
                    image1 = Image.open(image1_file)
                    st.image(image1, caption="Image 1", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image 1: {e}")
        
        with col2:
            st.subheader("Potential Dupe")
            image2_file = st.file_uploader(
                "Upload second image",
                type=['png', 'jpg', 'jpeg'],
                key="img2"
            )
            if image2_file is not None:
                try:
                    image2 = Image.open(image2_file)
                    st.image(image2, caption="Image 2", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image 2: {e}")
        
        # Compare button
        if image1_file is not None and image2_file is not None:
            if st.button("🔍 Compare Images", use_container_width=True):
                with st.spinner("Comparing images..."):
                    # Get classifications
                    pred1, conf1, _ = mock_predict_class(image1, class_names)
                    pred2, conf2, _ = mock_predict_class(image2, class_names)
                    
                    # Compute similarity
                    similarity = mock_compute_similarity(image1, image2)
                    is_duplicate = is_dupe(similarity, threshold)
                
                # Display results
                st.markdown("---")
                st.subheader("Comparison Results")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Image 1 Category", pred1, f"{conf1:.1f}%")
                
                with col_b:
                    st.metric("Similarity Score", f"{similarity:.1f}%", 
                             "✅ DUPE!" if is_duplicate else "❌ Not Dupe")
                
                with col_c:
                    st.metric("Image 2 Category", pred2, f"{conf2:.1f}%")
                
                # Verdict
                if is_duplicate:
                    st.success(
                        f"🎯 **DUPE DETECTED!** These items have {similarity:.1f}% similarity "
                        f"(above {threshold}% threshold)"
                    )
                else:
                    st.warning(
                        f"⚠️ **Not Dupes.** Similarity is {similarity:.1f}% "
                        f"(below {threshold}% threshold)"
                    )
                
                # Similarity gauge
                st.subheader("Similarity Gauge")
                fig, ax = plt.subplots(figsize=(10, 2))
                
                # Create gradient bar
                gradient = np.linspace(0, 100, 256).reshape(1, -1)
                ax.imshow(gradient, aspect='auto', cmap='RdYlGn', 
                         extent=[0, 100, 0, 1])
                
                # Add similarity marker
                ax.axvline(similarity, color='blue', linewidth=3, 
                          label=f'Similarity: {similarity:.1f}%')
                ax.axvline(threshold, color='red', linewidth=2, 
                          linestyle='--', label=f'Threshold: {threshold}%')
                
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Similarity Score (%)')
                ax.set_yticks([])
                ax.legend(loc='upper right')
                ax.set_title('Similarity Score Visualization')
                
                st.pyplot(fig)
        else:
            st.info("Please upload both images to compare them")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Fashion Dupe Detection System | Demo Mode | Powered by Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
