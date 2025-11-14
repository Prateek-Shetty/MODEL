from ultralytics import YOLO
import cv2

model_path = r"e:\MODEL\face_detection\face_detection.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()
    cv2.imshow("Webcam Face Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
