import cv2
from ultralytics import YOLO

# 1. Load the model (We're using YOLOv8m here, but you can swap this with your 'best.pt' once fully trained)
model_path = r"D:\Github\Hackathons\KAAL\MODEL\yolov8m.pt"
print(f"Loading model: {model_path}")
model = YOLO(model_path)

# 2. Specify the directory containing the images you want to test
source_dir = r"D:\Github\Hackathons\KAAL\MODEL\Datasets\military-artillery-and-tanks\2nd part int\2nd part int"

print(f"Running inference on directory: {source_dir}...")

# 3. Run the prediction
results = model.predict(
    source=source_dir,
    conf=0.25,        # Minimum confidence threshold stringency
    iou=0.45,         # Removes overlapping duplicate boxes
    show=False,       # Disabled UI popup since we are processing a large folder
    save=True,        # Saves the output image
    project=r"D:\Github\Hackathons\KAAL\MODEL\Processed_Outputs",
    name="military_artillery_2nd_part",    # Final folder: Processed_Outputs\military_artillery_2nd_part
    exist_ok=True     # Overwrites if folder exists
)

# 4. How to process the raw output data yourself
print("\n--- RESULTS DATA ---")
for result in results:
    # results comprises a list of objects (one for each image processed)
    boxes = result.boxes
    print(f"Found {len(boxes)} objects in the image.\n")
    
    for box in boxes:
        # Extract Box Data
        cls_id = int(box.cls[0].item())            # Class ID (0-based index)
        conf = float(box.conf[0].item())           # Confidence score (0.0 to 1.0)
        xyxy = box.xyxy[0].cpu().numpy()           # Coordinates [x1, y1, x2, y2]
        
        # Get actual class name from the model's dictionary
        class_name = model.names[cls_id]
        
        print(f"➤ Object:  {class_name}")
        print(f"  Conf:    {conf * 100:.1f}%")
        print(f"  Coords:  [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]\n")

print("Processing complete! Check the D:\\Github\\Hackathons\\KAAL\\MODEL\\Processed_Outputs\\military_artillery_2nd_part folder for saved images.")
