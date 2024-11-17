import requests
from urllib.request import urlopen
import base64

def test_single_image():
    # Get test image
    url = 'https://storage.googleapis.com/kagglesdsdata/datasets/1272/2280/testSample/testSample/img_1.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241115%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241115T154043Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=58e5c79ba897aa871408b7d061a84aafde1fbc1cce780ead1a50c8011547f311e6346b1da264d937f9ba4fc8ce46c902ef0021784df75cdee71954c32b7c5fd9637d20d40f4dfc0392977b9aae98b568412f073fe9480a8f8f851986e504e68907f62f0922829b34c79b522ec65ff21af64ef01e0e4c2f0b88e20d4ca3bf63bb3b4557276179454326a801a39ff443b781a4f956375c61b0240a28d8336989d445b0b7e266d2cd799afcaf3c5c033454bc4ba7b3f38f0c1b881bf293bdfe22028ad25d9f263d43597004fb9bb2fa5f363c2eae5976487d629f7d75a018b43e14abd552e02dfe36204508e4f6fdae7179ad603c095d6ce4298fe2c124643375ba'
    img_data = urlopen(url).read()
    
    # Convert to base64 string
    img_bytes = base64.b64encode(img_data).decode('utf-8')
    # Send request
    response = requests.post(
        "http://localhost:8000/predict",
        json={"image": img_bytes}
    )
    
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        print("\nPredictions:")
        for pred in predictions:
            print(f"Digit {pred['digit']}: {pred['probability']:.2%}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_single_image()