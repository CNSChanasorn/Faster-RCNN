import streamlit as st
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# นำเข้าฟังก์ชันจากไฟล์ prediction.py
from prediction import predict_and_draw

# Set title 
st.title('Cancer Cell Classification & Detection')
st.header('Please upload a picture')

@st.cache_resource
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    
    # กำหนดจำนวน Class (รวม Background)
    num_classes = 4 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load('faster_rcnn_best.pth', map_location=torch.device('cpu'))
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model

model = load_model()

# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # *** สำคัญ: เปลี่ยนชื่อ Class ให้ตรงกับข้อมูลเซลล์มะเร็งที่คุณใช้เทรน ***
    # Index 0 สำหรับ Faster R-CNN ต้องเป็น 'Background' เสมอครับ
    class_names = ['Background', 'apoptosis', 'normal', 'uncertain']

    if st.button('Prediction'):
        st.write("กำลังประมวลผล...")
        
        # ส่งไปประมวลผลที่ prediction.py
        fig, class_counts = predict_and_draw(model, image, class_names, threshold=0.5)
        
        # แสดงภาพที่วาดกรอบ Bounding Box แล้ว
        st.write("## Detection Image")
        st.pyplot(fig)
        
        # แสดงผลสรุปจำนวนคล้ายๆ กับแอปเดิม
        st.write("## Prediction Result")
        total_cells = sum(class_counts.values())
        st.write(f"**Total Cells Detected: {total_cells}**")
        
        # ลูปแสดงผลจำนวนของแต่ละ Class
        for class_name, count in class_counts.items():
            # ถ้าตรวจเจอเซลล์ชนิดนั้น ให้แสดงตัวหนังสือสีแดง (หรือเปลี่ยนสีได้ตามต้องการ)
            color = "red" if count > 0 else "black"
            st.write(f"### <span style='color:{color}'>{class_name} : {count} cells</span>", unsafe_allow_html=True)