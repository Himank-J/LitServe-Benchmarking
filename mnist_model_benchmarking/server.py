import torch
from PIL import Image
import io
import litserve as ls
import base64
import numpy as np
from torchvision import transforms

precision = torch.bfloat16

class MNISTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the model and necessary components"""
        self.device = device
        print(f"Using device: {device}")
        
        # Load the MNIST model
        state_dict = torch.load('model_checkpoint/model.pt', map_location=self.device)
        self.model = MNISTModel()  
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).to(precision) # half precision
        self.model.eval()

        # Replace the transforms with torchvision.transforms
        self.transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  
            transforms.Resize((28, 28)),                  
            transforms.ToTensor(),                       
        ])

        # MNIST labels (0-9)
        self.labels = list(range(10))
        print("Model loaded...")
    
    def decode_request(self, request):
        """Convert base64 encoded image to tensor"""
        image_bytes = request.get("image")
        if not image_bytes:
            raise ValueError("No image data provided")
        return image_bytes

    def batch(self, inputs):
        """Process and batch multiple inputs"""
        batched_tensors = []
        for image_bytes in inputs:
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(image_bytes)
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(img_bytes))
            # Transform image to tensor
            tensor = self.transforms(image)
            batched_tensors.append(tensor)
            
        # Stack all tensors into a batch
        return torch.stack(batched_tensors).to(self.device).to(precision)

    @torch.no_grad()
    def predict(self, x):
        outputs = self.model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities

    def encode_response(self, output):
        """Convert model output to API response"""
        # Handle both single and batched predictions
        if output.dim() == 1:
            probs, indices = torch.topk(output, k=1)
        else:
            probs, indices = torch.topk(output, k=1, dim=1)
        
        return {
            "predictions": [
                {
                    "digit": idx.item(),
                    "probability": prob.item()
                }
                for idx, prob in zip(indices.flatten(), probs.flatten())
            ]
        }

if __name__ == "__main__":
    api = ImageClassifierAPI()
    # Configure server with batching
    server = ls.LitServer(
        api,
        accelerator="gpu",
        max_batch_size=64,  
        batch_timeout=0.01,
        workers_per_device=4
    )
    server.run(port=8000)