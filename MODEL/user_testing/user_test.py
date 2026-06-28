import cv2
from ultralytics import YOLO

# 1. Load the model (We're using YOLOv8m here, but you can swap this with your 'best.pt' once fully trained)
model_path = r"D:\Github\Hackathons\KAAL\MODEL\yolov8m.pt"
print(f"Loading model: {model_path}")
model = YOLO(model_path)

# 2. Specify the image you want to test (You can also pass a web-cam ID like '0', or a video file path)
# I've grabbed one of your aircraft test images as an example!
image_path = r"D:\teest\1.jpeg"

print(f"Running inference on: {image_path}...")

# 3. Run the prediction
results = model.predict(
    source=image_path,
    conf=0.25,        # Minimum confidence threshold
    iou=0.45,         # Removes overlapping duplicate boxes
    show=False,       # Set to False so it doesn't block execution if you're not at the console
    save=True,        # Saves the output image
    project=r"D:\Github\Hackathons\KAAL\MODEL\runs\detect\runs\robustness\adv_training_run3\weights\runs",
    name="detect",    # The final folder will be weights\runs\detect
    exist_ok=True     # Overwrite if folder exists
)

# 4. How to extract and process the raw output data yourself
print("\n--- RESULTS DATA ---")
for result in results:
    boxes = result.boxes
    print(f"Found {len(boxes)} objects in the image.\n")
    
    for box in boxes:
        # Extract Box Data
        cls_id = int(box.cls[0].item())            # Class ID (0-based index)
        conf = float(box.conf[0].item())           # Confidence score (0.0 to 1.0)
        xyxy = box.xyxy[0].cpu().numpy()           # Coordinates [x1, y1, x2, y2]
        
        class_name = model.names[cls_id]           # Get actual class name from the model's dictionary
        
        print(f"➤ Object:  {class_name}")
        print(f"  Conf:    {conf * 100:.1f}%")
        print(f"  Coords:  [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]\n")

print(f"\nImage successfully saved to: {results[0].save_dir}")
print("Finished!")
