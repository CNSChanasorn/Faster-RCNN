import os
import sys
import gdown
import streamlit as st
from PIL import Image

# Suppress cv2 import issues
import warnings
warnings.filterwarnings('ignore')

# นำเข้าฟังก์ชันจากไฟล์ prediction.py
from prediction import predict_and_draw

# Set title 
st.title('Cancer Cell Object Detection')
st.header('Please upload a picture')

@st.cache_resource
def load_model():
    # Lazy import to avoid cv2 issues on startup
    import os
    os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
    from ultralytics import YOLO
    
    model_path = 'yolo12n_best.pt'
    
    # 1. เช็กว่าไฟล์โมเดลมีอยู่ในระบบ Streamlit Cloud หรือยัง
    if not os.path.exists(model_path):
        st.warning("กำลังดาวน์โหลดโมเดลขนาดใหญ่ โปรดรอสักครู่... (ทำแค่ครั้งแรก)")
        
        # 2. นำ File ID ที่ได้จาก Step 1 มาใส่ตรงนี้
        file_id = '1aisVcXuQJMHxIzg-BTiwUoM27SDrY4iL'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # 3. สั่งดาวน์โหลด
        gdown.download(url, model_path, quiet=False)
        st.success("ดาวน์โหลดโมเดลสำเร็จ!")

    # 4. โหลดโครงสร้างโมเดล Faster R-CNN (โค้ดเดิมของคุณ)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 4 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 5. โหลดน้ำหนัก (Weights)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
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