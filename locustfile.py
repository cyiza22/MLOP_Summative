"""Locust load testing"""
from locust import HttpUser, task, between
import numpy as np
from PIL import Image
import io


class MLAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def predict_image(self):
        """Test prediction endpoint"""
        # Create random test image
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {'image': ('test.png', img_bytes, 'image/png')}
        self.client.post("/predict", files=files)
    
    @task(1)
    def health_check(self):
        """Test health endpoint"""
        self.client.get("/health")
    
    @task(1)
    def get_metrics(self):
        """Test metrics endpoint"""
        self.client.get("/metrics")