from cProfile import label
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox


# Функция для выбора видеофайла
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
    if file_path:
        label.config(text=f"Выбран файл: {file_path}")
        process_video(file_path)
    else:
        messagebox.showwarning("Предупреждение", "Файл не выбран!")


# Функция для обработки видео
def process_video(video_path):
    # Проверка наличия файлов YOLO
    if not os.path.isfile('yolov3.weights') or not os.path.isfile('yolov3.cfg') or not os.path.isfile('coco.names'):
        messagebox.showerror("Ошибка", "Не найдены необходимые файлы YOLO: yolov3.weights, yolov3.cfg или coco.names")
        return

    # Загрузка YOLO модели
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

    # Загрузка названий классов
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Функция для обработки и предсказания объектов на каждом кадре
    def process_frame(frame, net, classes):
        height, width = frame.shape[:2]

        # Преобразование изображения для подачи в нейронную сеть YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Получение слоев выходных данных YOLO
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # Получение предсказаний сети
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # Обработка каждого выхода нейронной сети
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Порог уверенности
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Координаты прямоугольника
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Применение не максимального подавления для удаления пересекающихся рамок
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Отрисовка рамок и подписей
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Зеленый цвет для рамки
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    # Загрузка видеофайла
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        messagebox.showerror("Ошибка", "Ошибка при открытии видеофайла!")
        return

    try:
        while cap.isOpened():
            # Чтение кадра с видео
            ret, frame = cap.read()
            if not ret:
                print("Видео завершено или произошла ошибка.")
                break

            # Обработка кадра для распознавания объектов
            processed_frame = process_frame(frame, net, classes)

            # Отображение кадра с распознанными объектами
            cv2.imshow('Object Detection on Video', processed_frame)

            # Остановка программы при нажатии клавиши 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Программа была прервана.")

    finally:
        # Освобождение ресурсов
        cap.release()
        cv2.destroyAllWindows()


# Графический интерфейс
root = tk.Tk()
root.title("Object Detection with YOLOv3")
root.geometry("500x200")

# # Надпись с инструкцией
# label = tk.Label(root, text="Выберите видеофайл для распознавания объектов", font=("Arial", 12))
# label.pack(pady=20)
#
# # Кнопка для в
# import tkinter as tk

# root = tk.Tk()
# root.title("Тест Tkinter")
# root.mainloop()
