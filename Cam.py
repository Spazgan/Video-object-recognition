import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


# Загрузка YOLO
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

# Обработка каждого кадра
def process_frame(frame, net, classes):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
#COMMENT
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


# Отображение результатов
def draw_labels(frame, boxes, confidences, class_ids, classes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Проверяем, является ли indexes одномерным массивом
    if len(indexes) > 0:
        indexes = indexes.flatten()  # Убедимся, что это одномерный массив
        for i in indexes:
            box = boxes[i]
            (x, y, w, h) = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Зеленый цвет для рамок
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Обработка видео
def process_video(file_path):
    net, classes = load_yolo()
    cap = cv2.VideoCapture(file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confidences, class_ids = process_frame(frame, net, classes)
        draw_labels(frame, boxes, confidences, class_ids, classes)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Выбор видеофайла через GUI
def select_file():
    file_path = filedialog.askopenfilename(title="Выберите видео файл",
                                           filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")))
    if file_path:
        process_video(file_path)


# Создание GUI
root = tk.Tk()
root.title("Обработка видео")
root.geometry("300x100")

btn = tk.Button(root, text="Выбрать видео", command=select_file)
btn.pack(pady=20)

root.mainloop()
