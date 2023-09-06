import os
import sys
import pathlib
module_path = str(pathlib.Path('__file__').resolve().parent.parent)
sys.path.append(module_path)

import importlib
import unittest
import glob
import inspect
from readoc import Pattern, CompositePattern, DocumentCache, document_cache, convert_hashable
from readoc.pdf import Rect, Cell, CellRelation, Table, RegexDict, PDFPattern, TextBoxPattern, MatrixTablePattern, ListTablePattern
from readoc.pdf import convert_to_img, convert_to_binary_img, convert_to_drawings_img, convert_to_characters_img, extract_word_boxes, extract_text, extract_tables
from readoc.pdf.patterns import _parse_position, _parse_cell_range, _find_match_table, _does_shape_match


def get_tests_from_module(module):
    all_tests = []
    test_cases = [cls for _, cls in inspect.getmembers(module, inspect.isclass)
                  if issubclass(cls, unittest.TestCase)]
    for test_case in test_cases:
        test_names = [test_name for test_name, _ in inspect.getmembers(test_case, lambda m: inspect.isfunction(m) or inspect.ismethod(m))
                      if test_name.startswith("test")]
        for test_name in test_names:
            all_tests.append(test_case(test_name))
    
    return all_tests


if __name__ == '__main__':
    suite = unittest.TestSuite()
    test_programs = glob.glob("./**/test_*.py", recursive=True)
    for program in test_programs:
        program = os.path.splitext(os.path.basename(program))[0]
        module = importlib.import_module(program)
        tests = get_tests_from_module(module)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
