from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

class YOLOBagDetector:
    def __init__(self, yolo_weights_path='best.pt', device="cpu"):
        self.model = YOLO(yolo_weights_path).to(device)

    def predict(self, video_path, n_frames=2, conf=0.5):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = os.path.splitext(video_path)[0] + "_result.mp4"
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        if not out.isOpened():
            raise RuntimeError("Не удалось создать выходное видео")

        frame_idx = 0
        unique_bag = set()

        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % n_frames == 0:
                    try:
                        results = self.model.track(frame, conf=conf, persist=True, verbose=False)[0]

                        if results.boxes.id is not None and len(results.boxes.id) > 0:
                            for box, track_id in zip(results.boxes.xyxy, results.boxes.id):
                                unique_bag.add(int(track_id))
                                x1, y1, x2, y2 = map(int, box.tolist())
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(frame, f"bag {int(track_id)}", (x1, y1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    except Exception as e:
                        print(f"Ошибка обработки кадра {frame_idx}: {e}")

                    cv2.putText(frame, f"count: {len(unique_bag)}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    out.write(frame)

                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()
        return out_path