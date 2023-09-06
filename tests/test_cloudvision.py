import os
import sys
import pathlib
module_path = str(pathlib.Path('__file__').resolve().parent.parent)
sys.path.append(module_path)

import unittest
import fitz
import numpy as np

from readoc import CompositePattern
from readoc.pdf import Rect
from readoc.pdf.cloudvision import GoogleVisionOCRPattern
from readoc.pdf.cloudvision import extract_word_boxes_google_vision_ocr, extract_text_google_vision_ocr


class Test_CloudVision(unittest.TestCase):
    '''Test of readoc.pdf.cloudvision module'''

    def setUp(self) -> None:
        np.random.seed(74111114)
        test_dir = os.path.dirname(__file__)
        test_pdf_file = os.path.join(test_dir, "priceindex.pdf")
        self.pdf = fitz.open(test_pdf_file)


    def test_OCR_functions(self):
        def find_word_box(word, word_boxes):
            for wb in word_boxes:
                if word in wb[1]:
                    return wb
            return None
        
        rr = 7
        word_boxes = extract_word_boxes_google_vision_ocr(self.pdf, 1, resolution_rate=rr)
        extracted_text = extract_text_google_vision_ocr(self.pdf, 1, Rect(35,708,193,725)*rr, resolution_rate=rr)

        self.assertIsNotNone(find_word_box("比較", word_boxes))
        self.assertIsNotNone(find_word_box("家事", word_boxes))
        self.assertIsNotNone(find_word_box("魚介", word_boxes))
        self.assertIsNotNone(find_word_box("からあげ", word_boxes))
        self.assertTrue((Rect(296,363,344,387)*rr).includes(find_word_box("豚肉", word_boxes)[0]))
        self.assertEqual(extracted_text, "ラスパイレス連鎖基準方式による指数")
    
    # def test_mpl(self):
    #     from matplotlib import pyplot as plt
    #     from matplotlib.font_manager import FontProperties
    #     fp = FontProperties(fname="/Users/ekohei/Downloads/ipaexg00401/ipaexg.ttf", size=16)
    #     def plot_square(ax=plt, bbox=(0,0,1,1), label=None):
    #         p1 = [bbox[0],bbox[1]]
    #         p2 = [bbox[2],bbox[1]]
    #         p3 = [bbox[2],bbox[3]]
    #         p4 = [bbox[0],bbox[3]]
    #         ps = np.array([p1,p2,p3,p4,p1])
    #         ax.plot(ps[:,0], ps[:,1])
    #         if label is not None:
    #             cx = (bbox[0] + bbox[2])/2
    #             cy = (bbox[1] + bbox[3])/2
    #             ax.text(cx, cy, label, ha="center", va="center", fontsize=6, fontproperties=fp)
    #         return
        
    #     word_boxes = extract_word_boxes_google_vision_ocr(self.pdf, 1)
    #     for wb in word_boxes:
    #         plot_square(plt, bbox=wb[0], label=wb[1])
    #     rect = self.pdf[1].rect
    #     plt.gca().set_xlim(0,rect.width)
    #     plt.gca().set_ylim(rect.height,0)
    #     plt.show()

    
    def test_GoogleVisionOCRPattern(self):
        pattern_data = {
            "patterns": [
                {
                    "class": "GoogleVisionOCRPattern",
                    "page": 0,
                    "position": [450,38,537,75],
                    "action": "check_format",
                    "expected_text": "総務省",
                    "exact_match": True,
                    "score_weight": 1
                },
                {
                    "class": "GoogleVisionOCRPattern",
                    "page": 0,
                    "position": [51,40,139,70],
                    "action": "extract_text",
                    "name": "title"
                },
                {
                    "class": "GoogleVisionOCRPattern",
                    "page": 0,
                    "position": [400,80,540,120],
                    "action": "check_format",
                    "expected_text": "令和5年6月23日",
                    "score_weight": 1
                },
                {
                    "class": "GoogleVisionOCRPattern",
                    "page": 0,
                    "position": [80,150,140,190],
                    "action": "check_format",
                    "expected_text": "概況",
                    "exact_match": True,
                    "score_weight": 1
                },
                {
                    "class": "TextBoxPattern",
                    "page": 0,
                    "position": [70,180,290,210],
                    "action": "extract_text",
                    "name": "abstract01"
                }
            ]
        }
        cp = CompositePattern(pattern_data)
        result = cp.process_document(self.pdf)
        self.assertEqual(result["check_count"], 11)
        self.assertEqual(result["match_score"], 1.0)
        self.assertEqual(result["value"]["title"], "報道資料")
        self.assertTrue("総合指数は2020年を100として105.1" in result["value"]["abstract01"])




if __name__ == "__main__":
    unittest.main()