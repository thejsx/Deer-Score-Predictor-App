import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

class AntlerNet(nn.Module):
    def __init__(self):
        super(AntlerNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 28 * 28, 512),  # Assuming the input is 224x224
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1)  # Outputting a single continuous score
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.squeeze(dim=1)

model = AntlerNet()
model.load_state_dict(torch.load("antler_net.pth", map_location=torch.device('cpu')))
model.eval()

def predict_value(image):
    """Resize, preprocess, and classify the image using the PyTorch model."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction.item()

st.title("Deer Image Score Predictor")
st.write("Upload an image to get an AI score prediction:")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("Predicting...")
    predicted_value = predict_value(image)
    st.write(f"Predicted value: {predicted_value:.2f}")