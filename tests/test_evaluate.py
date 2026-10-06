import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from src.evaluate import run_evaluation
from src.common import read_json

class EvaluationTests(unittest.TestCase):
    def test_image_root_scoring_and_no_overwrite(self):
        class Engine:
            def predict(self, image, question):
                return {'answer':'24340','ocr_text':'ZIP 24340'}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            Image.new('RGB',(12,12),'white').save(root/'image.png')
            rows=[{'question_id':1,'image_path':'image.png','question':'ZIP?', 'answers':['24340']}]
            output=root/'results.json'
            with patch('src.evaluate.sync_gpu'):
                summary=run_evaluation(Engine(),rows,root,output,'downsample50')
            self.assertEqual(summary['correct'],1)
            self.assertEqual(read_json(output)[0]['transformation'],'downsample50')
            with self.assertRaises(FileExistsError):
                run_evaluation(Engine(),rows,root,output)
