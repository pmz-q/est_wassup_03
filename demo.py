import streamlit as st
from PIL import Image
import numpy as np
from demo_tools import get_emotion, make_cropped_image

uploaded_file = st.file_uploader(label='원본 이미지를 업로드해주세요', type=['png', 'jpg', 'jpeg'])

cropped = None;
result = None;

if uploaded_file is not None:
  cropped = make_cropped_image(np.array(Image.open(uploaded_file)))

if cropped is not None:
  result = get_emotion(cropped)


col1, col2, col3 = st.columns([3,4,4]);

if uploaded_file is not None:
    with col1:
      st.image(uploaded_file)

if cropped is not None:
    with col2:
      st.image('demo_tools/output_cropped.jpg')

if result is not None:
    with col3:
      st.image('demo_tools/output_emotion.jpg')
