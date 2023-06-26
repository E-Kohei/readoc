import os
import sys
sys.path.append("/Users/ekohei/Programming/python")

import json
import unittest
import fitz
import numpy as np
from readoc import Pattern, CompositePattern, DocumentCache, document_cache, convert_hashable
from readoc.pdf import Rect, Cell, CellRelation, Table, PDFPattern, TextBoxPattern, MatrixTablePattern, ListTablePattern


class DictPattern(Pattern):
    def __init__(self, check_number=False, **kwargs):
        super().__init__(**kwargs)
        self.check_number = check_number

    @document_cache(share_with_other_patterns=True)
    def get_rawdata(self, document, **kwargs):
        rawdata = document["rawdata"]
        pairs = rawdata.split(",")
        for i,p in enumerate(pairs):
            pairs[i] = tuple(p.split(":"))
        return dict(pairs)

    def process_document(self, document):
        super().process_document(document)
        check_count = 3 if self.check_number else 2
        match_count = 0
        value = {}
        doc = document
        if "text1" in doc:
            match_count += 1
        if "text2" in  doc:
            match_count += 1
        if self.check_number and "number" in doc:
            match_count += doc["number"]
        if "rawdata" in doc:
            rawdata = self.get_rawdata(document, additional_vain_argument="hello")
            value.update(rawdata)
        return {
            "check_count": check_count,
            "match_score": match_count,
            "value": value
        }


class Test_Core(unittest.TestCase):
    '''Test of readoc.core'''

    def test_Pattern(self):
        mock_document = {
            "title": "Hello, world!",
            "number": 100,
            "text1": "Welcome to Readoc!",
            "text2": "Readoc supports extracting (semi-)structured data from unstructured data.",
            "rawdata": "foo:12,bar:50,baz:100"
        }
        pattern_data = {
            "patterns": [
                {
                    "class": "DictPattern",
                    "check_number": True,
                    "score_weight": 10,
                    "vain_property": "foo"
                }
            ]
        }
        composite_pattern = CompositePattern(pattern_data)
        result = composite_pattern.process_document(mock_document)

        self.assertEqual(result["check_count"], 3)
        self.assertAlmostEqual(result["match_score"], 102.0)
        self.assertEqual(result["value"]["foo"], "12")
    

    def test_cache(self):
        cache = DocumentCache.get_instance()
        cache.clear()
        cache2 = DocumentCache.get_instance()
        self.assertTrue(cache is cache2)

        mock_document = {
            "title": "Hello, world!",
            "number": 100,
            "text1": "Welcome to Readoc!",
            "text2": "Readoc supports extracting (semi-)structured data from unstructured data.",
            "rawdata": "foo:12,bar:50,baz:100"
        }
        dp = DictPattern()
        rd = dp.get_rawdata(mock_document, additional_vain_argument="hello")
        _ = dp.get_rawdata(mock_document, additional_vain_argument="goodbye")

        key1 = ('get_rawdata', ('document', (('title', 'Hello, world!'), ('number', 100), ('text1', 'Welcome to Readoc!'), ('text2', 'Readoc supports extracting (semi-)structured data from unstructured data.'), ('rawdata', 'foo:12,bar:50,baz:100'))), ('kwargs', (('additional_vain_argument', 'hello'),)))
        self.assertEqual(cache.get(key1), {'foo': '12', 'bar': '50', 'baz': '100'})
        key2 = ('get_rawdata', ('document', (('title', 'Hello, world!'), ('number', 100), ('text1', 'Welcome to Readoc!'), ('text2', 'Readoc supports extracting (semi-)structured data from unstructured data.'), ('rawdata', 'foo:12,bar:50,baz:100'))), ('kwargs', (('additional_vain_argument', 'goodbye'),)))
        self.assertEqual(cache.get(key2), {'foo': '12', 'bar': '50', 'baz': '100'})
        self.assertFalse(cache.get(key1) is rd)
        self.assertEqual(len(cache), 2)
    

    def test_convert_hashable(self):
        lst = [1, 2, 3, [10,20], {"foo": {"bar": 100}}]
        hashable_lst = convert_hashable(lst)
        self.assertEqual(hashable_lst, ( 1, 2, 3, (10, 20), (('foo', (('bar', 100),)),) ))




class Test_PDF(unittest.TestCase):
    '''Test of readoc.pdf module'''

    def setUp(self) -> None:
        np.random.seed(74111114)
        test_pdf_file = os.path.join(os.getcwd(), "test", "priceindex.pdf")
        self.pdf = fitz.open(test_pdf_file)


    def test_Rect(self):
        r1 = Rect(0,10,100,50)
        r2 = Rect([0,20,70,40])
        self.assertEqual(r1.height(), 40)
        self.assertTrue(r1.includes(r2))

    def test_Cell(self):    
        def rrect(base_rect):
            return Rect(base_rect) + Rect(np.random.random(4)*5-2.5)
        
        c11 = Cell(rrect((20,10,70,60)), "City");      c12 = Cell(rrect((70,10,170,60)), "Pok*mon center"); c13 = Cell(rrect((170,10,270,60)), "Jim")
        c21 = Cell(rrect((20,60,70,140)), "Aspertia"); c22 = Cell(rrect((70,60,270,140)), "Yes")
        c31 = Cell(rrect((20,140,70,300)), "Nacrene"); c32 = Cell(rrect((70,140,170,220)), "Yes");          c33 = Cell(rrect((170,140,270,220)), "Yes")
        pass;                                          c42 = Cell(rrect((70,220,170,300)), "Yes");          c43 = Cell(rrect((170,220,270,300)), "No")
        c_ext = Cell((15,5,275,305), "")
        
        self.assertEqual(c11.bbox, c11["bbox"])
        self.assertTrue(95 <= c12.width() <= 105)
        self.assertTrue(45 <= c12.height() <= 55)
        self.assertTrue(c_ext.includes(c22))
        self.assertFalse(c11.includes(c12))
        
        self.assertEqual(CellRelation.infer_cell_relation(c11, c12), CellRelation.SAME_ROW)
        self.assertEqual(CellRelation.infer_cell_relation(c11, c31), CellRelation.SAME_COL)
        self.assertEqual(CellRelation.infer_cell_relation(c22, c43), CellRelation.PARTIALLY_SAME_COL)
        self.assertEqual(CellRelation.infer_cell_relation(c32, c22), CellRelation.PARTIALLY_SAME_COL)
        self.assertEqual(CellRelation.infer_cell_relation(c31, c32), CellRelation.PARTIALLY_SAME_ROW)
        self.assertEqual(CellRelation.infer_cell_relation(c42, c31), CellRelation.PARTIALLY_SAME_ROW)
    
    def test_Table(self):
        def rrect(base_rect):
            return Rect(base_rect) + Rect(np.random.random(4)*5-2.5)
        
        c11 = Cell(rrect((20,10,70,60)), "City");      c12 = Cell(rrect((70,10,170,60)), "Pok*mon center"); c13 = Cell(rrect((170,10,270,60)), "Jim")
        c21 = Cell(rrect((20,60,70,140)), "Aspertia"); c22 = Cell(rrect((70,60,270,140)), "Yes")
        c31 = Cell(rrect((20,140,70,300)), "Nacrene"); c32 = Cell(rrect((70,140,170,220)), "Yes");          c33 = Cell(rrect((170,140,270,220)), "Yes")
        pass;                                          c42 = Cell(rrect((70,220,170,300)), "Yes");          c43 = Cell(rrect((170,220,270,300)), "No")
        c_ext = Cell((15,5,275,305), "")
        
        table = Table.from_cells([c11, c12, c13, c22, c31, c32, c33, c42, c43])
        self.assertEqual(table.shape, (4,3))
        self.assertTrue(table.is_row_merged_cell(3,0))
        self.assertTrue(table.is_col_merged_cell(1,2))
        self.assertTrue(245 <= table.get_width() <= 255)
        self.assertTrue(285 <= table.get_height() <= 295)
        self.assertTrue(95 <= table.get_width_of_col(1) <= 105)
        self.assertTrue(75 <= table.get_height_of_row(3) <= 85)
    
    def test_PDFPattern(self):
        from matplotlib import pyplot as plt
        cache = DocumentCache.get_instance()
        cache.clear()

        pp = PDFPattern()
        pdf_img  = pp.convert_to_img(self.pdf, page=0, resolution_rate=3)
        scaled_rect = self.pdf[0].rect * 3
        self.assertAlmostEqual(pdf_img.shape[0], scaled_rect[3], delta=1)
        self.assertAlmostEqual(pdf_img.shape[1], scaled_rect[2], delta=1)
        pdf_bin_img = pp.convert_to_binary_img(self.pdf, page=0, binary_threshold=200, resolution_rate=3)
        self.assertEqual(len(cache), 2)

        drawings_img = pp.convert_to_drawings_img(self.pdf, page=0, character_size_threshold="auto")
        characters_img = pp.convert_to_characters_img(self.pdf, 0, dilate_kernel=np.ones((8,8), np.uint8))
        self.assertEqual(len(cache), 6)
        # plt.imshow(drawings_img, cmap="gray")
        # plt.show()

        word_boxes = pp.extract_word_boxes(self.pdf, 0)
        self.assertTrue(word_boxes[0][1] == "令")

        extracted_text = pp.extract_text(self.pdf, 0, [150,100,430,140])
        self.assertEqual(extracted_text, "2020年基準消費者物価指数")
        t = pp.extract_text(self.pdf, 0, [400,80,540,120])
        print(t)

        pos1 = pp.calc_position(self.pdf, 0, [0, 1000, 100, 500])
        pos2 = pp.calc_position(self.pdf, 0, [100, '*', 100, '*'])
        pos3 = pp.calc_position(self.pdf, 0, [0.0, 0.5, 1.0, 0.8])
        pos4 = pp.calc_position(self.pdf, 0, [0.1, 0.2, '*', 1.0], 5)

        rect = self.pdf[0].rect
        self.assertEqual(pos1, Rect(0,1000,100,500))
        self.assertEqual(pos2, Rect(100,-np.inf,100,np.inf))
        self.assertEqual(pos3, Rect(0, rect[1]*0.5, rect[2], rect[3]*0.8))
        self.assertEqual(pos4, Rect(rect[0]*0.1*5, rect[1]*0.2*5, np.inf, rect[3]*5))

    def test_TextBoxPattern(self):
        pattern_data = {
            "patterns": [
                {
                    "class": "TextBoxPattern",
                    "page": 0,
                    "position": [400,80,540,120],
                    "action": "check_format",
                    "expected_text": "令 和 ５ 年 ６ 月 23 日",
                    "score_weight": 1
                },
                {
                    "class": "TextBoxPattern",
                    "page": 0,
                    "position": [80,150,140,190],
                    "action": "check_format",
                    "expected_text": "概況",
                    "score_weight": 1
                },
                {
                    "class": "TextBoxPattern",
                    "page": 0,
                    "position": [70,180,290,210],
                    "action": "extract_text",
                    "target_label": "abstract01"
                }
            ]
        }
        cp = CompositePattern(pattern_data)
        result = cp.process_document(self.pdf)
        self.assertEqual(result["check_count"], 2)
        self.assertEqual(result["match_score"], 1.0)
        self.assertTrue("総合指数は2020年を100として105.1" in result["value"]["abstract01"])
    
    def test_MatrixTablePattern(self):
        pass
    
    def tearDown(self) -> None:
        self.pdf.close()
    
#     def test_PDFPattern(self):
#         pdd = PDfDocumentData()



if __name__ == '__main__':
    # file = "./billing1.pdf"
    # doc = fitz.open(file)
    # page = doc[0]
    # dd = DocumentData(page)
    # # The above line should be as the following in the future release.
    # # dd = DocumentData(doc)
    
    
    unittest.main()