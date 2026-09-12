"""
Automated Integration Test Suite for AgriEdge Intelligence Application.
Tests ML Inference, Flask REST Endpoints, Schedulers, and DB DAOs in-memory.
"""

import os
import sys
import unittest
import json
from io import BytesIO
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app
from scheduler.adaptive_scheduler import compare_all_schedulers
from memory.memory_manager import memory_manager
from deadlock.deadlock_arbitrator import simulate_deadlock_scenario

class TestAgriEdgeOS(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_dashboard_summary_api(self):
        res = self.client.get('/api/dashboard_summary')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('summary', data)
        self.assertIn('weather', data)
        self.assertIn('system_resources', data)
        print("✓ Dashboard Summary API Passed")

    def test_translation_api(self):
        res = self.client.get('/api/translate/ta')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('btn_upload_image'), 'இலை புகைப்படத்தைப் பதிவேற்று')
        print("✓ Tamil Translation API Passed")

    def test_compare_schedulers_api(self):
        res = self.client.get('/api/compare_schedulers')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('Adaptive', data['comparison'])
        self.assertIn('FCFS', data['comparison'])
        print("✓ Scheduler Benchmark API Passed")

    def test_deadlock_check_api(self):
        res = self.client.get('/api/deadlock_check?scenario=safe')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['is_safe'])
        print("✓ Deadlock Arbitration (Banker's Algorithm) API Passed")

    def test_farmer_assistant_api(self):
        res = self.client.post('/api/farmer_assistant', json={'query': 'What fertilizer is suitable?', 'language': 'en'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('Nutrient Management', data['category'])
        print("✓ Farmer Assistant Panel API Passed")

    def test_disease_inference(self):
        # Generate dummy leaf image
        img = Image.new('RGB', (224, 224), color='green')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        data = {
            'image': (img_bytes, 'test_leaf.jpg'),
            'model': 'efficientnet_b0'
        }
        res = self.client.post('/api/analyze_disease', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        result = res.get_json()
        self.assertIn('crop', result)
        self.assertIn('disease', result)
        self.assertIn('confidence', result)
        self.assertIn('severity', result)
        self.assertIn('actions', result)
        print("✓ PyTorch EfficientNet-B0 Disease Inference API Passed:", result['crop'], "-", result['disease'])

if __name__ == '__main__':
    unittest.main()
