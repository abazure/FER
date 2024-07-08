import numpy as np # type: ignore
import cv2 as cv # type: ignore
from yunet import YuNet
from ultralytics import YOLO # type: ignore
import streamlit as st # type: ignore

# Check OpenCV version

# Valid combinations of backends and targets
# backend_target_pairs = [
#     [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
#     [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
#     [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
#     [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
#     [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
# ]

def visualize(image, results, expressions, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for (det, expr) in zip(results, expressions):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        label = f"{expr[0]} ({expr[1]:.2f})"
        cv.putText(output, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    return output

def main():
    st.title('Face Expression Recognition App')

    st.sidebar.title('Settings')
    input_type = st.sidebar.radio('Input Type', ['Image', 'Webcam'])

    # model_path = st.sidebar.text_input('Model Path', 'face_detection_yunet_2023mar.onnx')
    model_path = 'face_detection_yunet_2023mar.onnx'
    # backend_target = st.sidebar.selectbox('Backend Target', list(range(len(backend_target_pairs))), format_func=lambda x: ['OpenCV + CPU', 'CUDA + GPU', 'CUDA + GPU (FP16)', 'TIM-VX + NPU', 'CANN + NPU'][x])
    conf_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.9)

    # backend_id = backend_target_pairs[backend_target][0]
    # target_id = backend_target_pairs[backend_target][1]
    backend_id = cv.dnn.DNN_BACKEND_OPENCV
    target_id = cv.dnn.DNN_TARGET_CPU

    if backend_id == cv.dnn.DNN_BACKEND_CANN:
        st.error("CANN backend is not available in the current OpenCV build. Please choose a different backend.")
        return

    # Instantiate YuNet for face detection
    face_model = YuNet(modelPath=model_path,
                       inputSize=[320, 320],
                       confThreshold=conf_threshold,
                       backendId=backend_id,
                       targetId=target_id)

    # Instantiate YOLO for expression recognition
    expr_model = YOLO('Yolov8m-cls-150-16/weights/best.pt')

    if input_type == 'Image':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv.imdecode(file_bytes, 1)
            h, w, _ = image.shape

            # Inference
            face_model.setInputSize([w, h])
            results = face_model.infer(image)

            # Process each detected face for expression recognition
            expressions = []
            for det in results:
                bbox = det[0:4].astype(np.int32)
                face_img = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                expr_results = expr_model(face_img)
                top_result_idx = expr_results[0].probs.top1
                top_result_label = expr_results[0].names[top_result_idx]
                confidence = expr_results[0].probs.top1conf.item()
                expressions.append((top_result_label, confidence))

            # Draw results on the input image
            image = visualize(image, results, expressions)

            # Display the result
            st.image(image, channels="BGR")
    else:
        run = st.checkbox('Run Webcam')
        FRAME_WINDOW = st.image([])
        camera = cv.VideoCapture(2)

        if not camera.isOpened():
            st.error("Error: Could not open video device.")
            return

        while run:
            has_frame, frame = camera.read()
            if not has_frame:
                st.warning("No frames captured.")
                continue

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            face_model.setInputSize([w, h])
            results = face_model.infer(frame)

            # Process each detected face for expression recognition
            expressions = []
            for det in results:
                bbox = det[0:4].astype(np.int32)
                face_img = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                expr_results = expr_model(face_img)
                top_result_idx = expr_results[0].probs.top1
                top_result_label = expr_results[0].names[top_result_idx]
                confidence = expr_results[0].probs.top1conf.item()
                expressions.append((top_result_label, confidence))

            # Draw results on the input image
            frame = visualize(frame, results, expressions)

            FRAME_WINDOW.image(frame)
        else:
            st.write('Stopped')
            camera.release()

if __name__ == '__main__':
    main()
