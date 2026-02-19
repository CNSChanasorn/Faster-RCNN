import os
import gdown
import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from prediction import predict_and_draw

st.title('Cancer Cell Object Detection')
st.header('Please upload a picture')

@st.cache_resource
def load_model():
    model_path = 'faster_rcnn_best.pth'
    
    if not os.path.exists(model_path):
        st.warning("กำลังดาวน์โหลดโมเดลขนาดใหญ่ โปรดรอสักครู่... (ทำแค่ครั้งแรก)")
        
        file_id = '1aisVcXuQJMHxIzg-BTiwUoM27SDrY4iL'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        gdown.download(url, model_path, quiet=False)
        st.success("ดาวน์โหลดโมเดลสำเร็จ!")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 4 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

model = load_model()

uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    class_names = ['Background', 'apoptosis', 'normal', 'uncertain']

    if st.button('Prediction'):
        st.write("Processing...")
        
        fig, class_counts = predict_and_draw(model, image, class_names, threshold=0.5)
        
        st.write("## Detection Image")
        st.pyplot(fig)
        
        st.write("## Prediction Result")
        total_cells = sum(class_counts.values())
        st.write(f"**Total Cells Detected: {total_cells}**")
        
        for class_name, count in class_counts.items():
            color = "red" if count > 0 else "black"
            st.write(f"### <span style='color:{color}'>{class_name} : {count} cells</span>", unsafe_allow_html=True)