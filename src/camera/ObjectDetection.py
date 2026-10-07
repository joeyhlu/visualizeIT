import cv2
import numpy as np
import time
from ultralytics import YOLO
from data import COCO_CLASSES

# Constants
WIDTH = 1280
HEIGHT = 720
AREA_THRESH = 30
CONFIDENCE = 0.15

class ObjectDetection:
    def __init__(self, capture_index=0, model_path="yolov8n-seg.pt"):
        self.width, self.height = WIDTH, HEIGHT
        self.capture_idx = capture_index
        self.model = YOLO(model_path)

    def get_all_detections(self, frame):
        resized_frame = cv2.resize(frame, (640, 640))
        results = self.model(resized_frame)

        res = results[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # shape [N,4]
        confs = res.boxes.conf.cpu().numpy()       # shape [N]
        classes = res.boxes.cls.cpu().numpy()      # shape [N]

        masks = res.masks.data.cpu().numpy() if res.masks is not None else None

        for i in list(classes):
            print(f"{i} and {COCO_CLASSES[int(i)]}")

        H, W = frame.shape[:2]
        boxes_scaled = []
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            x1 = int(x1 * W / 640)
            y1 = int(y1 * H / 640)
            x2 = int(x2 * W / 640)
            y2 = int(y2 * H / 640)
            boxes_scaled.append([x1, y1, x2, y2])
        boxes_scaled = np.array(boxes_scaled, dtype=np.int32)

        return boxes_scaled, confs, classes, masks
    
    def find_bounding_box(self, frame, target_cls):
        resized_frame = cv2.resize(frame, (640, 640))
        results = self.model(resized_frame)

        res = results[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # shape [N, 4]
        confs = res.boxes.conf.cpu().numpy()       # shape [N]
        classes = res.boxes.cls.cpu().numpy()      # shape [N]

        H, W = frame.shape[:2]

        for i, cls in enumerate(classes):
            if int(cls) == target_cls:
                # Scale the bounding box to the original frame size
                x1, y1, x2, y2 = boxes_xyxy[i]
                x1 = int(x1 * W / 640)
                y1 = int(y1 * H / 640)
                x2 = int(x2 * W / 640)
                y2 = int(y2 * H / 640)

                return (x1, y1, x2, y2), confs[i]

        return None

    def run(self):
        video = cv2.VideoCapture(self.capture_idx)
        if not video.isOpened():
            print("Camera not found or can't be opened.")
            return
        video.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while True:
            ret, frame = video.read()
            if not ret:
                break
            ti = time.time()
            boxes, confs, classes, masks = self.get_all_detections(frame)
            
            fps = 1 / (time.time() - ti)

            for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, classes):
                if conf < CONFIDENCE: 
                    continue
                cls_name = COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else f"Class {int(cls)}"
                label = f"{cls_name} {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label text
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
                )
            
            cv2.putText(
                img=frame,
                text=f'FPS: {int(fps)}',
                org=(20, 70),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=(227, 111, 179),
                thickness=2
            )

            cv2.imshow("Detection with Labels", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection(0)
    detector.run()
