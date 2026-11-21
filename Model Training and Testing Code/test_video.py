
from ultralytics import YOLO

# Load your trained model
model = YOLO("new_best.pt")

# Process the video
results = model.predict(
    source="test_videos/helmet_video.mp4",
    save=True,
    conf=0.10,
    device="cpu"
)

print("Video processing completed!")
print("Check the runs/detect folder for your output video.")
