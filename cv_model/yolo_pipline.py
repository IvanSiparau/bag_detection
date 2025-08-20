from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
import torch

class YOLOBagDetector:
    def __init__(self, yolo_weights_path='best.pt', device="cpu"):
        if torch.cuda.is_available():
            try:
                self.model = YOLO(yolo_weights_path).to('cuda')
            except Exception as e:
                self.model = YOLO(yolo_weights_path).to('cpu')
        else:
            self.model = YOLO(yolo_weights_path).to('cpu')
        self.left_line = [1, 325, 204, 130]
        self.right_line = [284, 348, 402, 147]
        self.cross_line = [116,250,223, 250]

    def _point_side_of_line(self, cx, cy):
        if cy < self.cross_line[1]:
            return "above"
        return "below"


    def _is_center_between_lines(self, cx, cy):
        def is_left_of_line(x, y, line):
            x1, y1, x2, y2 = line
            return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) > 0

        return is_left_of_line(cx, cy, self.left_line) and not is_left_of_line(cx, cy, self.right_line)
    
    def _is_center_inside_box(self, cx, cy, box):
        return box[0] < cx < box[2] and box[1] < cy < box[3]

    def predict(self, video_path, n_frames=2, conf=0.6):
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
        prev_positions = {}
        count_out = 0

        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % n_frames == 0:
                    try:
                        results = self.model.track(frame, conf=conf, persist=True, verbose=False)[0]

                        if results.boxes.id is not None and len(results.boxes.id) > 0:
                            bag_list = []
                            for box, track_id in zip(results.boxes.xyxy, results.boxes.id):
                                x1, y1, x2, y2 = map(int, box.tolist())
                                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                                if self._is_center_between_lines(cx, cy):
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                    cv2.putText(frame, f"bag {int(track_id)}", (x1, y1 - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    side = self._point_side_of_line(cx, cy)
                                    prev_side = prev_positions.get(int(track_id))

                                    if prev_side == "below" and side == "above":
                                        if count_out == 0:
                                            bag_list.append([
                                                int(track_id),
                                                [x1, y1, x2, y2]
                                            ])
                                        else:
                                          count_out -= 1

                                    elif side == "below" and prev_side == "above":
                                        count_out += 1

                                    prev_positions[int(track_id)] = side
                            
                            bags_to_add = []

                            for bag in bag_list:
                                track_id, box = bag
                                cx = (box[0] + box[2]) / 2
                                cy = (box[1] + box[3]) / 2
                                is_nested = False
                                for other_bag in bag_list:
                                    if bag[0] != other_bag[0]:
                                        if self._is_center_inside_box(cx, cy, other_bag[1]):
                                            is_nested = True
                                            break
                                
                                if not is_nested and track_id not in unique_bag:
                                    bags_to_add.append(track_id)

                            for track_id in bags_to_add:
                                unique_bag.add(track_id)
          

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