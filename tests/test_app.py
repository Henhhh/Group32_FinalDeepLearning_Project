import io
import unittest
from unittest.mock import patch
from PIL import Image
try:
    import flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

@unittest.skipUnless(FLASK_AVAILABLE, 'Flask not installed')
class AppTests(unittest.TestCase):
    def setUp(self):
        from app import create_app
        class FakeEngine:
            def predict(self, image, question):
                return {'answer':'60.000','ocr_text':'TOTAL 60.000'}
        self.app = create_app('unused', engine_factory=lambda _: FakeEngine())
        self.client = self.app.test_client()
    def test_home(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code,200)
        self.assertIn('Document question answering'.encode(),response.data)
    def test_requires_input(self):
        self.assertEqual(self.client.post('/api/answer').status_code,400)
    def test_upload_and_answer(self):
        raw=io.BytesIO();Image.new('RGB',(8,8),'white').save(raw,format='PNG');raw.seek(0)
        with patch('app.sync_gpu'):
            r=self.client.post('/api/answer',data={'image':(raw,'receipt.png'),'question':'Total?','method':'ocr'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json['answer'],'60.000')
        self.assertIn('seconds',r.json)
    def test_invalid_image(self):
        r=self.client.post('/api/answer',data={'image':(io.BytesIO(b'bad'),'bad.png'),'question':'Total?'})
        self.assertEqual(r.status_code,400)
