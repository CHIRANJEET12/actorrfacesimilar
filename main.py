import streamlit as st
from predict import predict_actor
from PIL import Image
import tempfile
import os
import cv2

st.set_page_config(page_title="Bollywood Look-Alike", layout="centered")

st.title("🎬 Bollywood Actor Look-Alike")
st.write("Upload your photo and see which Bollywood actor you resemble!")

# Add tips for users
with st.expander("💡 Tips for best results"):
    st.write("""
    1. Upload a clear, front-facing photo
    2. Make sure your face is well-lit
    3. Remove sunglasses or hats
    4. Look directly at the camera
    5. Ensure your full face is visible
    """)

uploaded_file = st.file_uploader(
    "Upload a clear face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Your Photo")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        # Convert RGBA to RGB if needed
        if image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            rgb_image.save(tmp.name, format="JPEG", quality=95)
        elif image.mode != "RGB":
            image = image.convert("RGB")
            image.save(tmp.name, format="JPEG", quality=95)
        else:
            image.save(tmp.name, format="JPEG", quality=95)
        
        temp_path = tmp.name
    
    with col2:
        st.subheader("Analysis")
        if st.button("🔍 Find My Look-Alike", type="primary", use_container_width=True):
            try:
                with st.spinner("Analyzing your face..."):
                    # Add progress indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Loading image...")
                    progress_bar.progress(10)
                    
                    # Try to predict
                    results = predict_actor(temp_path, top_k=5)
                    
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    if results:
                        st.success("✅ Analysis Complete!")
                        st.balloons()
                        
                        # Display top matches in a nice way
                        st.subheader("🎯 Your Top Matches")
                        
                        # Create columns for results
                        cols = st.columns(len(results))
                        
                        for idx, result in enumerate(results):
                            with cols[idx]:
                                # Display actor image
                                try:
                                    actor_img = Image.open(result['image_path'])
                                    st.image(actor_img, caption=result['actor'], use_container_width=True)
                                except:
                                    st.warning(f"Could not load image for {result['actor']}")
                                
                                # Display similarity score with color coding
                                score = result['similarity']
                                if score >= 70:
                                    color = "green"
                                    emoji = "🎯"
                                elif score >= 50:
                                    color = "orange"
                                    emoji = "👍"
                                else:
                                    color = "red"
                                    emoji = "👀"
                                
                                st.markdown(
                                    f"""
                                    <div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px;'>
                                        <h3 style='color: {color}; margin: 0;'>{score}%</h3>
                                        <p style='margin: 0; font-size: 14px;'>{emoji} Match</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                        
                        # Show detailed results
                        st.subheader("📊 Detailed Results")
                        for result in results:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.write(f"**{result['actor']}**")
                            with col2:
                                score = result['similarity']
                                if score >= 70:
                                    st.success(f"{score}%")
                                elif score >= 50:
                                    st.warning(f"{score}%")
                                else:
                                    st.error(f"{score}%")
                            with col3:
                                if score >= 70:
                                    st.write("Strong match! 👑")
                                elif score >= 50:
                                    st.write("Good match! ✨")
                                else:
                                    st.write("Some resemblance")
                    
                    else:
                        st.warning("No matches found. Try with a different image.")
                        
            except Exception as e:
                error_msg = str(e)
                st.error("❌ " + error_msg)
                
                # Provide specific guidance based on error
                if "Face not detected" in error_msg or "Could not extract" in error_msg:
                    st.info("""
                    **Tips to improve face detection:**
                    - Try a clearer photo with better lighting
                    - Make sure your face is fully visible
                    - Look directly at the camera
                    - Avoid extreme angles
                    - Try without glasses if possible
                    """)
                    
                    # Show example of good image
                    with st.expander("See example of a good photo"):
                        st.write("A good photo should have:")
                        st.write("- Clear visibility of face features")
                        st.write("- Good lighting (not too dark or bright)")
                        st.write("- Front-facing angle")
                        st.write("- Neutral expression")
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)