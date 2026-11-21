
from ultralytics import YOLO

# Load your already-trained model
model = YOLO("new_best.pt")

# Test YOLO on pictures in the test_images folder
results = model.predict(
    source="test_images",
    save=True,
    conf=0.25,
    device="cpu"
)

print("YOLO testing completed!")
print("Your detected images have been saved.")
