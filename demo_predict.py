
from PIL import Image
from ultralytics import YOLO
# Load a pretrained YOLO11n model
model = YOLO("D:/ultralytics-main/runs/detect/train13/weights/best.pt")  # load a custom model
#model = YOLO("yolov8n.pt")  # load a custom model
# Run inference on 'bus.jpg'
results = model("D:/ultralytics-main/datasets/medicine/test/images/20240828_094423_757_157.bmp")  # predict on an image
#results = model("ultralytics/assets/bus.jpg")  # predict on an image
# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")