import os
import torch
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import draw_bounding_boxes

# Load pretrained Faster R-CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=weights
)

model = model.to(device)
model.eval()

# COCO class names
categories = weights.meta["categories"]

print("Device:", device)
print("Number of classes:", len(categories))

# Use sample food image or upload
image_path = "sample_food.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = to_tensor(image).to(device)

print("Image size:", image.size)

# Perform object detection
with torch.no_grad():
    prediction = model([image_tensor])[0]

# Confidence threshold
threshold = 0.5
keep = prediction["scores"] >= threshold

boxes = prediction["boxes"][keep].cpu()
labels = prediction["labels"][keep].cpu()
scores = prediction["scores"][keep].cpu()

print("Detected objects:")
for label, score in zip(labels, scores):
    print(f"{categories[label]}: {score:.2f}")

# Draw bounding boxes
image_for_display = to_tensor(image).mul(255).byte()

box_labels = [
    f"{categories[label]}: {score:.2f}"
    for label, score in zip(labels.tolist(), scores.tolist())
]

annotated_image = draw_bounding_boxes(
    image_for_display,
    boxes=boxes,
    labels=box_labels,
    colors="red",
    width=3
)

plt.figure(figsize=(12, 8))
plt.imshow(annotated_image.permute(1, 2, 0))
plt.axis("off")
plt.title("Food Object Detection using Faster R-CNN")
plt.savefig("chit12_plot.png")
plt.close()

print("Object detection completed successfully!")
print("Chit 12 completed successfully!")
