import os
from datetime import datetime 
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_SAVE_PHOTOS = r"workspace\training_demo\saves_photo"

os.makedirs(PATH_TO_SAVE_PHOTOS, exist_ok=True)

PATH_TO_LABELS = r"C:\Users\Nike\Desktop\Scripts\Python\TenserFlow\workspace\training_demo\annotations\label_map.pbtxt"
#'annotations/label_map.pbtxt'

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
tf.get_logger().setLevel('ERROR')         

PATH_TO_CFG = r"C:\Users\Nike\Desktop\Scripts\Python\TenserFlow\workspace\training_demo\exported-models\my_model\pipeline.config"
#'exported-models/my_model1/pipeline.config'

configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

PATH_TO_CKPT = r"C:\Users\Nike\Desktop\Scripts\Python\TenserFlow\workspace\training_demo\exported-models\my_model\checkpoint" 
#'exported-models/my_model1/checkpoint/' тут на папку путь а не на скрипт
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

cap = cv2.VideoCapture(0)

while True:
    ret, image_np = cap.read()
    image_np_expanded = np.expand_dims(image_np, axis=0)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.80,
          agnostic_mode=False)

    detection_boxes = detections['detection_boxes'][0].numpy() # Кординаты рамок
    detection_classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int) #Что конкретно нашли
    detection_scores = detections['detection_scores'][0].numpy() # Процент уверености в обнаружении

    save_image = False

    for i in range(detection_boxes.shape[0]):
        if detection_scores[i] >= 0.90: #фильтр по порогу обнаружения
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
            cv2.imwrite(output_path, image_np_with_detections)
            print(f"Изображение сохранено как {output_path}")
            save_image = False

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
