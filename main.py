import streamlit as st
from predict import predict_actor
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Bollywood Look-Alike", layout="centered")

st.title("🎬 Bollywood Actor Look-Alike")
st.write("Upload your photo and see which Bollywood actor you resemble!")

uploaded_file = st.file_uploader(
    "Upload a clear face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Your Photo", width="stretch")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image = Image.open(uploaded_file)
        if image.mode == "RGBA":
            rgb_image = Image.new("RGB", image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        elif image.mode != "RGB":
            image = image.convert("RGB")

        image.save(tmp.name, format="JPEG", quality=95)
        temp_path = tmp.name

    if st.button("Find My Look-Alike 🎭", type="primary"):
        with st.spinner("Analyzing face..."):
            try:
                results = predict_actor(temp_path, top_k=5)

                st.success("Analysis Complete!")
                st.subheader("🎯 Your Top Matches")

                cols = st.columns(len(results))

                for idx, result in enumerate(results):
                    with cols[idx]:
                        actor_img = Image.open(result['image_path'])
                        st.image(actor_img, caption=result['actor'], use_container_width=True)

                        score = result['similarity']
                        if score > 80:
                            st.markdown(f"<h4 style='color: green;'>{score}%</h4>", unsafe_allow_html=True)
                        elif score > 60:
                            st.markdown(f"<h4 style='color: orange;'>{score}%</h4>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h4 style='color: red;'>{score}%</h4>", unsafe_allow_html=True)

                st.subheader("📊 Detailed Results")
                for result in results:
                    st.write(f"**{result['actor']}** - {result['similarity']}% similarity")

            except Exception as e:
                st.error("Face not detected. Please upload a clear image with a visible face.")

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)