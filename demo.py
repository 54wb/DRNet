import os
import clip
import torch
from PIL import Image

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Load Fair classes
CLASSES = ('Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A321', 'A220', 
            'A330', 'A350', 'C919', 'ARJ21', 'other-airplane', 'Passenger_Ship', 
            'Motorboat', 'Fishing_Boat', 'Tugboat', 'Engineering_Ship', 'Liquid_Cargo_Ship', 'Dry_Cargo_Ship', 
            'Warship', 'other-ship', 'Small_Car', 'Bus', 'Cargo_Truck', 'Dump_Truck', 'Van', 'Trailer', 'Tractor', 'Truck_Tractor', 
            'Excavator', 'other-vehicle', 'Baseball_Field', 'Basketball_Court', 'Football_Field', 'Tennis_Court', 'Roundabout', 
            'Intersection', 'Bridge')

# Prepare the inputs
image_path = "/disk0/lwb/datasets/Fair1m1_0/train/images/2305.tif"
image = Image.open(image_path)
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in CLASSES]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{CLASSES[index]:>16s}: {100 * value.item():.2f}%")