import os
import sys
import pathlib
module_path = str(pathlib.Path('__file__').resolve().parent.parent)
sys.path.append(module_path)

import unittest
import fitz
import numpy as np

from readoc import CompositePattern, DocumentCache
from readoc.pdf import Rect, Cell, CellRelation, Table, RegexDict, PDFPattern, TextBoxPattern, MatrixTablePattern, ListTablePattern
from readoc.pdf import convert_to_img, convert_to_binary_img, convert_to_drawings_img, convert_to_characters_img, extract_word_boxes, extract_text, extract_tables
from readoc.pdf.patterns import _parse_position, _parse_cell_range, _find_match_table, _does_shape_match


class Test_PDF(unittest.TestCase):
    '''Test of readoc.pdf module'''

    def setUp(self) -> None:
        np.random.seed(74111114)
        test_dir = os.path.dirname(__file__)
        test_pdf_file = os.path.join(test_dir, "priceindex.pdf")
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
    
    def test_RegexDict(self):
        import re
        rdict = RegexDict()
        rdict["foo,bar,baz"] = 12345
        rdict[("aaa", "bbb", "ccc")] = "hello"
        rdict[("aaa", "bbb", "ddd")] = "goodbye"

        self.assertEqual(rdict["foo,bar,baz"], 12345)
        pattern = re.compile("foo*")
        self.assertEqual(rdict[pattern], [12345])
        self.assertEqual(rdict[r"foo,bar,baz"], 12345)
        self.assertEqual(rdict.rget(r"foo*"), [12345])
        self.assertTrue(rdict.is_rin("aaa", "bbb"))
        self.assertEqual(rdict.rget("aaa", "bbb"), ["hello", "goodbye"])

    
    def test_PDF_functions(self):
        from matplotlib import pyplot as plt
        def plot_square(ax=plt, bbox=(0,0,1,1), label=None):
            p1 = [bbox[0],bbox[1]]
            p2 = [bbox[2],bbox[1]]
            p3 = [bbox[2],bbox[3]]
            p4 = [bbox[0],bbox[3]]
            ps = np.array([p1,p2,p3,p4,p1])
            ax.plot(ps[:,0], ps[:,1])
            if label is not None:
                cx = (bbox[0] + bbox[2])/2
                cy = (bbox[1] + bbox[3])/2
                ax.text(cx, cy, label, ha="center", va="center", fontsize=6, fontproperties=fp)
            return

        cache = DocumentCache.get_instance()
        cache.clear()

        pdf_img  = convert_to_img(self.pdf, page=0, resolution_rate=3)
        scaled_rect = self.pdf[0].rect * 3
        self.assertAlmostEqual(pdf_img.shape[0], scaled_rect[3], delta=1)
        self.assertAlmostEqual(pdf_img.shape[1], scaled_rect[2], delta=1)

        pdf_bin_img = convert_to_binary_img(self.pdf, page=0, binary_threshold=200, resolution_rate=3)
        self.assertEqual(len(cache), 2)

        drawings_img = convert_to_drawings_img(self.pdf, page=0, character_size_threshold="auto")
        characters_img = convert_to_characters_img(self.pdf, 0, dilate_kernel=np.ones((8,8), np.uint8))
        self.assertEqual(len(cache), 6)
        # plt.imshow(drawings_img, cmap="gray")
        # plt.show()

        word_boxes = extract_word_boxes(self.pdf, 0)
        self.assertTrue(word_boxes[0][1] == "令")

        extracted_text = extract_text(self.pdf, 0, [150,100,430,140])
        self.assertEqual(extracted_text, "2020年基準消費者物価指数")

        extracted_tables = extract_tables(self.pdf, page=1, binary_threshold=180, resolution_rate=7.0)
        self.assertEqual(len(extracted_tables), 3)
        self.assertEqual(extracted_tables[0].shape, (4,15))

        rect = self.pdf[0].rect
        pos1 = _parse_position([0, 1000, 100, 500], rect)
        pos2 = _parse_position([100, '*', 100, '*'], rect)
        pos3 = _parse_position([0.0, 0.5, 1.0, 0.8], rect)
        pos4 = _parse_position([0.1, 0.2, '*', 1.0], rect, resolution_rate=5)
        self.assertEqual(pos1, Rect(0,1000,100,500))
        self.assertEqual(pos2, Rect(100,-np.inf,100,np.inf))
        self.assertEqual(pos3, Rect(0, rect[1]*0.5, rect[2], rect[3]*0.8))
        self.assertEqual(pos4, Rect(rect[0]*0.1*5, rect[1]*0.2*5, np.inf, rect[3]*5))

        cell_range1 = _parse_cell_range([[1,2],[9,10]], (10,12))
        cell_range2 = _parse_cell_range([(1,2),(-1,-2)], (10,12))
        self.assertEqual(cell_range1, cell_range2)
        self.assertTrue((9,10) in cell_range1)
        self.assertFalse((9,11) in cell_range1)
        self.assertTrue((5,7) in cell_range1)
        self.assertFalse((1,1) in cell_range1)

        c11 = Cell((20,10,70,60), "");   c12 = Cell((70,10,170,60), "");   c13 = Cell((170,10,270,60), "")
        c21 = Cell((20,60,70,140), "");  c22 = Cell((70,60,270,140), "");  pass
        c31 = Cell((20,140,70,300), ""); c32 = Cell((70,140,170,220), ""); c33 = Cell((170,140,270,220), "")
        pass;                            c42 = Cell((70,220,170,300), ""); c43 = Cell((170,220,270,300), "")
        d11 = Cell((150,200,250,250), ""); d12 = Cell((250,200,350,250), ""); d13 = Cell((350,200,450,250), ""); d14 = Cell((450,200,550,250), "")
        d21 = Cell((150,250,250,300), ""); d22 = Cell((250,250,350,300), ""); pass;                              d24 = Cell((450,250,550,300), "")
        table1 = Table.from_cells([c11, c12, c13, c22, c31, c32, c33, c42, c43])
        table2 = Table.from_cells([d11, d12, d13, d14, d21, d22, d24])
        tables = [table1, table2]
        matched_table, actual_shape = _find_match_table(tables, Rect([100,190,600,310]), (2,'*'))
        self.assertTrue(matched_table is table2)
        self.assertEqual(actual_shape, (2,4))

        self.assertTrue(_does_shape_match((3,'*'), (3,10)))
        self.assertTrue(_does_shape_match(('*','*'), [10,12]))
    
    def test_TextBoxPattern(self):
        pattern_data = {
            "patterns": [
                {
                    "class": "TextBoxPattern",
                    "page": 0,
                    "position": [400,80,540,120],
                    "action": "check_format",
                    "expected_text": "令和５年６月23日",
                    "score_weight": 1
                },
                {
                    "class": "TextBoxPattern",
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
        self.assertEqual(result["check_count"], 10)
        self.assertEqual(result["match_score"], 1.0)
        self.assertTrue("総合指数は2020年を100として105.1" in result["value"]["abstract01"])
    
    def test_MatrixTablePattern(self):
        rr = 7
        landmarks = [
            {
                "loc": [1,0],
                "name": "total",
                "expected_text": "総合"
            }
        ]
        table_headers = [
            {
                "loc": [0,2],
                "name": "2022/05",
                "scope-range": [[1,2],[-1,2]],
                "expected_text": "５月"
            },
            {
                "loc": [0,3],
                "name": "2022/06",
                "scope-range": [[1,3],[-1,3]],
                "expected_text": "６月"
            },
            {
                "loc": [0,-1],
                "name": "2023/05",
                "scope-range": [[1,-1],[-1,-1]],
                "expected_text": "５月"
            },
            {
                "loc": [1,0],
                "name": "total",
                "scope-range": [[1,2],[1,-1]],
                "expected_text": "総合"
            },
            {
                "loc": [1,1],
                "name": "ratio_to_last_year",
                "scope-range": [[1,2],[1,-1]],
                "expected_text": "前年同月比(％)"
            },
            {
                "loc": [2,0],
                "name": "total_without_fresh_foods",
                "scope-range": [[2,2],[2,-1]],
                "expected_text": "生鮮食品を除く総合"
            },
            {
                "loc": [2,1],
                "name": "ratio_to_last_year",
                "scope-range": [[1,2],[1,-1]],
                "expected_text": "前年同月比(％)"
            },
        ]
        additional_info_cells = [
            {
                "loc": [3,0],
                "name": "total_without_fresh_foods_and_energy",
            }
        ]
        mtp = MatrixTablePattern(1, Rect([14,729,515,803])*rr, [4,15], landmarks=landmarks, table_headers=table_headers, additional_info_cells=additional_info_cells, resolution_rate=rr)
        result = mtp.process_document(self.pdf)
        
        self.assertEqual(result["check_count"], 1 + len(landmarks) + len(table_headers))
        self.assertEqual(result["match_score"], 1.0)

        result_rdict = RegexDict(result["value"])
        self.assertTrue(result_rdict.is_rin("total", "ratio_to_last_year"))
        self.assertEqual(result_rdict["total_without_fresh_foods_and_energy"], "生鮮食品及びエネルギーを除く総合")
        self.assertEqual(result_rdict.rget("total_without_fresh_foods", "2022/05"), ["2.0"])
        self.assertEqual(result_rdict.rget("total_without_fresh_foods", "2023/05"), ["3.4"])
    
    def test_ListTablePattern(self):
        rr = 7
        landmarks = [
            {
                "loc": [0,0],
                "name": "season_adjusted_value",
                "expected_text": "原数値"
            }
        ]
        table_headers = [
            {
                "loc": [0,0],
                "name": "month",
                "scope-range": [[0,2],[0,-1]],
                "expected_text": "原数値"
            },
            {
                "loc": [1,0],
                "name": "total",
                "scope-range": [[1,2],[2,-1]],
                "expected_text": "総合"
            },
            {
                "loc": [1,1],
                "name": "index",
                "scope-range": [[1,2],[1,-1]],
                "expected_text": "指数"
            },
            {
                "loc": [2,1],
                "name": "ratio_to_last_year",
                "scope-range": [[2,2],[2,-1]],
                "expected_text": "前年同月比(％)"
            },
            {
                "loc": [3,0],
                "name": "total_without_fresh_foods",
                "scope-range": [[3,2],[4,-1]],
                "expected_text": "生鮮食品を除く総合"
            },
            {
                "loc": [3,1],
                "name": "index",
                "scope-range": [[3,2],[3,-1]],
                "expected_text": "指数"
            },
            {
                "loc": [4,1],
                "name": "ratio_to_last_year",
                "scope-range": [[4,2],[4,-1]],
                "expected_text": "前年同月比(％)"
            },
            {
                "loc": [5,0],
                "name": "total_without_fresh_foods_and_energy",
                "scope-range": [[5,2],[6,-1]],
                "expected_text": "生鮮食品及びエネルギーを除く総合"
            },
            {
                "loc": [5,1],
                "name": "index",
                "scope-range": [[5,2],[5,-1]],
                "expected_text": "指数"
            },
            {
                "loc": [6,1],
                "name": "ratio_to_last_year",
                "scope-range": [[6,2],[6,-1]],
                "expected_text": "前年同月比(％)"
            },
        ]
        ltp = ListTablePattern(0, Rect([60,500,540,670])*rr, (7,15), grouping="col", landmarks=landmarks, table_headers=table_headers, additional_info_cells=[], resolution_rate=rr)
        result = ltp.process_document(self.pdf)

        # from matplotlib import pyplot as plt
        # i1 = convert_to_img(self.pdf, 0, resolution_rate=rr)
        # print(i1[3943:3948,1174:1177])
        # i2 = convert_to_binary_img(self.pdf, 0, binary_threshold=25, resolution_rate=rr)
        # plt.imshow(i2, cmap="gray"); plt.show()

        self.assertEqual(result["check_count"], 1 + len(landmarks) + len(table_headers))
        self.assertEqual(result["match_score"], 1.0)

        items = result["value"]["items"]
        self.assertEqual(len(items), 13)
        first_elem = {
            ("month",): "５月",
            ("total", "index"): "101.8",
            ("total", "ratio_to_last_year"): "2.5",
            ("total_without_fresh_foods", "index"): "101.6",
            ("total_without_fresh_foods", "ratio_to_last_year"): "2.1",
            ("total_without_fresh_foods_and_energy", "index"): "100.1",
            ("total_without_fresh_foods_and_energy", "ratio_to_last_year"): "0.8"
        }
        self.assertEqual(items[0], first_elem)
        self.assertEqual(items[-1][("month",)], "５月")
        self.assertEqual(items[-1][("total", "index")], "105.1")
    
    def test_pattern(self):
        test_dir = os.path.dirname(__file__)
        pattern_file = os.path.join(test_dir, "priceindex_pattern.json")
        cp = CompositePattern.open(pattern_file)
        result = cp.process_document(self.pdf)
        value = result["value"]
        print(result["match_score"])
        self.assertEqual(value["items"][0][("total","index")], "101.8")
        self.assertEqual(value["items"][2][("total_without_fresh_foods_and_energy", "ratio_to_last_year")], "1.2")
        self.assertEqual(value[("energy","2023/04","contribution_rate")], "-0.37")
        self.assertEqual(value[("kerosene","2023/05","ratio_to_last_month")], "-0.1")
    
    
    def tearDown(self) -> None:
        self.pdf.close()



if __name__  == '__main__':
    unittest.main()