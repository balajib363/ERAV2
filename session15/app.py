import gradio as gr

from PIL import Image
import numpy as np
import torch
import os
import cv2

# Load fine-tuned custom model
model = torch.hub.load('WongKinYiu/yolov9', 'custom', 'weights/best.pt',
                        force_reload=True, trust_repo=True)

# Declaring some variables    
TABLE_CONFIDENCE = 0.30
CELL_CONFIDENCE = 0.30

# Bounding Boxes color scheme
ALPHA = 0.2
TABLE_BORDER = (0, 0, 255)
CELL_FILL = (0, 0, 200)
CELL_BORDER = (0, 0, 255)

def inf_image(img_path):
    # Run the Inference and draw predicted bboxes
    results = model(img_path)
    df = results.pandas().xyxy[0]
    table_bboxes = []
    cell_bboxes = []
    for _, row in df.iterrows():
        if row['class'] == 0 and row['confidence'] > TABLE_CONFIDENCE:
            table_bboxes.append([int(row['xmin']), int(row['ymin']),
                                 int(row['xmax']), int(row['ymax'])])

        if row['class'] == 1 and row['confidence'] > CELL_CONFIDENCE:
            cell_bboxes.append([int(row['xmin']), int(row['ymin']),
                                int(row['xmax']), int(row['ymax'])])
    image = cv2.imread(img_path)
    overlay = image.copy()
    for table_bbox in table_bboxes:
        cv2.rectangle(image, (table_bbox[0], table_bbox[1]),
                      (table_bbox[2], table_bbox[3]), TABLE_BORDER, 2)

    for cell_bbox in cell_bboxes:
        cv2.rectangle(overlay, (cell_bbox[0], cell_bbox[1]),
                      (cell_bbox[2], cell_bbox[3]), CELL_FILL, -1)
        cv2.rectangle(image, (cell_bbox[0], cell_bbox[1]),
                      (cell_bbox[2], cell_bbox[3]), CELL_BORDER, 2)

    image_new = cv2.addWeighted(overlay, ALPHA, image, 1-ALPHA, 0)
    img = cv2.cvtColor(image_new,cv2.COLOR_BGR2RGB)
    return img


title = "YoloV9 Custom model Object detection"
description = "Detecting on images EXIT signboard"
examples = [["images/sample.jpg"], 
            ["images/sample_1.jpg"],
            ["images/sample_2.jpg"]]
app = gr.Interface(
    fn=inf_image,
    inputs=gr.Image(label="Input Image Component", type="filepath"),
    outputs=gr.Image(label="Output Image Component", type="numpy"),
    title = title,
    description = description,
    examples = examples,
    allow_flagging='manual',
)


app.launch()