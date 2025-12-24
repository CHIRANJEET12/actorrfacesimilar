import streamlit as st
from predict import predict_actor
from PIL import Image
import tempfile
import os
import time

st.set_page_config(page_title="Bollywood Look-Alike", layout="centered")

st.title("🎬 Bollywood Actor Look-Alike")
st.write("Upload your photo and see which Bollywood actor you resemble!")

# Add tips for users
with st.expander("💡 Tips for best results", expanded=False):
    st.write("""
    1. Upload a clear, front-facing photo
    2. Make sure your face is well-lit
    3. Remove sunglasses or hats
    4. Look directly at the camera
    5. Ensure your full face is visible
    """)

uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Photo")
            # Convert uploaded file to PIL Image
            image = Image.open(uploaded_file)
            # Resize for display
            max_size = (300, 300)
            image.thumbnail(max_size)
            st.image(image, caption="Uploaded Image", width=250)
        
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            # Handle different image modes
            if image.mode in ('RGBA', 'LA', 'P'):
                # Convert RGBA to RGB
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                rgb_image.save(tmp_file.name, 'JPEG', quality=95)
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                image.save(tmp_file.name, 'JPEG', quality=95)
            else:
                # Save original image
                image.save(tmp_file.name, 'JPEG', quality=95)
            
            temp_path = tmp_file.name
        
        with col2:
            st.subheader("Analysis")
            
            if st.button("🔍 Find My Look-Alike", type="primary"):
                # Create a progress container
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Step 1: Check file
                        status_text.text("Checking image file...")
                        progress_bar.progress(10)
                        time.sleep(0.5)
                        
                        # Step 2: Analyze image
                        status_text.text("Analyzing facial features...")
                        progress_bar.progress(30)
                        
                        # Call the prediction function
                        results = predict_actor(temp_path, top_k=5)
                        
                        progress_bar.progress(80)
                        status_text.text("Finding closest matches...")
                        time.sleep(0.5)
                        
                        progress_bar.progress(100)
                        status_text.text("Complete!")
                        time.sleep(0.5)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display results
                        if results:
                            st.success("✅ Analysis Complete!")
                            
                            # Show top match first
                            top_match = results[0]
                            st.subheader(f"🎯 Your Closest Match: **{top_match['actor']}**")
                            
                            col_img, col_info = st.columns([1, 2])
                            
                            with col_img:
                                try:
                                    actor_img = Image.open(top_match['image_path'])
                                    actor_img.thumbnail((200, 200))
                                    st.image(actor_img, caption=top_match['actor'], width=150)
                                except:
                                    st.warning("Could not load actor image")
                            
                            with col_info:
                                score = top_match['similarity']
                                if score >= 75:
                                    st.markdown(f"<h1 style='color: green;'>{score}% Match</h1>", unsafe_allow_html=True)
                                    st.write("**Strong resemblance!** 🎉")
                                elif score >= 50:
                                    st.markdown(f"<h1 style='color: orange;'>{score}% Match</h1>", unsafe_allow_html=True)
                                    st.write("**Good match!** ✨")
                                else:
                                    st.markdown(f"<h1 style='color: #ff6b6b;'>{score}% Match</h1>", unsafe_allow_html=True)
                                    st.write("**Some resemblance** 👀")
                            
                            # Show other matches
                            if len(results) > 1:
                                st.subheader("🎭 Other Possible Matches")
                                
                                # Create columns for other matches
                                cols = st.columns(len(results) - 1)
                                
                                for idx, result in enumerate(results[1:], 1):
                                    with cols[idx - 1]:
                                        try:
                                            actor_img = Image.open(result['image_path'])
                                            actor_img.thumbnail((150, 150))
                                            st.image(actor_img, caption=result['actor'], width=120)
                                        except:
                                            st.write(f"**{result['actor']}**")
                                        
                                        score = result['similarity']
                                        if score >= 70:
                                            st.markdown(f"<div style='color: green; text-align: center;'><b>{score}%</b></div>", unsafe_allow_html=True)
                                        elif score >= 50:
                                            st.markdown(f"<div style='color: orange; text-align: center;'><b>{score}%</b></div>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"<div style='color: #ff6b6b; text-align: center;'><b>{score}%</b></div>", unsafe_allow_html=True)
                            

                        else:
                            st.warning("No matches found. Please try with a different image.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        
                        if "Face not detected" in str(e) or "face" in str(e).lower():
                            st.info("""
                            **Tips to improve face detection:**
                            - Upload a clearer photo with better lighting
                            - Make sure your entire face is visible
                            - Try a front-facing photo
                            - Avoid dark backgrounds or shadows on your face
                            """)
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image file.")