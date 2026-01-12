from ultralytics import YOLO
model = YOLO("human.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("human.engine")