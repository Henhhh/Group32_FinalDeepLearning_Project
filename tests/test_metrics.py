import unittest
from src.metrics import calculate_anls, normalize_em, summarize
from src.common import degrade_image
from PIL import Image

class MetricTests(unittest.TestCase):
    def test_partial_credit_numeric_error(self):
        self.assertAlmostEqual(calculate_anls('14340', ['24340']), 0.8)
        self.assertNotEqual(normalize_em('14340'), normalize_em('24340'))
    def test_best_reference_and_threshold(self):
        self.assertEqual(calculate_anls('budget', ['the budget','Budget']), 1)
        self.assertEqual(calculate_anls('ab', ['ac']), 0)
        self.assertEqual(calculate_anls('xyz', ['abc']), 0)
    def test_em_and_anls_have_different_whitespace_rules(self):
        self.assertEqual(normalize_em(' A  B '), 'a b')
        self.assertLess(calculate_anls('a  b', ['a b']), 1)
    def test_summary_uses_all_references(self):
        result = summarize([{'prediction':'budget','answers':['the budget','Budget'],'seconds':2}])
        self.assertEqual(result['correct'],1)
        self.assertEqual(result['anls'],1)
    def test_degradation_preserves_canvas_but_changes_pixels(self):
        image = Image.new('RGB',(20,20))
        for x in range(20):
            for y in range(20):
                image.putpixel((x,y), ((x+y)%2*255,)*3)
        out = degrade_image(image)
        self.assertEqual(out.size,image.size)
        self.assertNotEqual(out.tobytes(),image.tobytes())

if __name__ == '__main__':
    unittest.main()
