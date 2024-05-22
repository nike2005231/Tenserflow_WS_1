import os
from datetime import datetime
import numpy as np
import cv2
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Подавление логов TensorFlow

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Путь для сохранения изображений
PATH_TO_SAVE_PHOTOS = r"workspace\training_demo\saves_photo"
os.makedirs(PATH_TO_SAVE_PHOTOS, exist_ok=True)

# Путь к файлу label_map.pbtxt
PATH_TO_LABELS = r"C:\Users\Nike\Desktop\Scripts\Python\TenserFlow\workspace\training_demo\annotations\label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
tf.get_logger().setLevel('ERROR')

# Путь к файлу pipeline.config
PATH_TO_CFG = r"C:\Users\Nike\Desktop\Scripts\Python\TenserFlow\workspace\training_demo\exported-models\my_model\pipeline.config"
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Путь к checkpoint
PATH_TO_CKPT = r"C:\Users\Nike\Desktop\Scripts\Python\TenserFlow\workspace\training_demo\exported-models\my_model\checkpoint"
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    """Обнаружение объектов на изображении."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections, prediction_dict, tf.reshape(shapes, [-1])

def start_camera():
    """Запуск камеры и обновление кадра."""
    ret, frame = cap.read()
    if not ret:
        return
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    frame_with_detections = frame.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.80,
        agnostic_mode=False)

    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)
    detection_scores = detections['detection_scores'][0].numpy()

    save_image = False

    for i in range(detection_boxes.shape[0]):
        if detection_scores[i] >= 0.90:
            box = detection_boxes[i]
            class_id = detection_classes[i]
            score = detection_scores[i]
            class_name = category_index[class_id]['name']

            print(f"Обнаружен {class_name} увереность: {score:.2f}")
            print(f"Кординаты разметки: {box}")

            save_image = True

        if save_image:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(PATH_TO_SAVE_PHOTOS, f"detected_image_{timestamp}.jpg")
            cv2.imwrite(output_path, frame_with_detections)
            print(f"Изображение сохранено как {output_path}")
            save_image = False

    # Преобразование изображения для отображения в QLabel
    frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    convert_to_qt_format = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    p = convert_to_qt_format.scaled(label.width(), label.height(), aspectRatioMode=Qt.KeepAspectRatio)
    label.setPixmap(QPixmap.fromImage(p))

# Запуск приложения
app = QApplication(sys.argv)

# Создание окна
window = QWidget()
window.setWindowTitle('Object Detection with TensorFlow and PyQt')
window.resize(1280, 720) 
layout = QVBoxLayout()

# Центрирование QLabel в окне
hbox = QHBoxLayout()
label = QLabel()
label.setMinimumSize(800, 600)  # Установка минимального размера для метки
hbox.addWidget(label, alignment=Qt.AlignCenter)  # Центрирование метки в горизонтальном направлении
layout.addLayout(hbox)

# Кнопка для запуска камеры
btn_start = QPushButton("Start Camera")
btn_start.setGeometry(50, 700, 200, 50) 
btn_start.move(100, 150)  
layout.addWidget(btn_start)

window.setLayout(layout)
window.show()

cap = cv2.VideoCapture(0)

# Таймер для обновления кадров
timer = QTimer()
timer.timeout.connect(start_camera)

def start_camera_feed():
    timer.start(1)

btn_start.clicked.connect(start_camera_feed)

sys.exit(app.exec_())
cap.release()
cv2.destroyAllWindows()