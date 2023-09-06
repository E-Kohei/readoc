import os
import sys
import pathlib
module_path = str(pathlib.Path('__file__').resolve().parent.parent)
sys.path.append(module_path)

import unittest

from readoc import Pattern, CompositePattern, DocumentCache, document_cache, convert_hashable


@document_cache
def get_rawdata(document, **kwargs):
    rawdata = document["rawdata"]
    pairs = rawdata.split(",")
    for i,p in enumerate(pairs):
        pairs[i] = tuple(p.split(":"))
    return dict(pairs)

class DictPattern(Pattern):
    def __init__(self, check_number=False, **kwargs):
        super().__init__(**kwargs)
        self.check_number = check_number

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
            rawdata = get_rawdata(document, additional_vain_argument="hello")
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



class Test_Cache(unittest.TestCase):

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
        rd = get_rawdata(mock_document, additional_vain_argument="hello")
        _ = get_rawdata(mock_document, additional_vain_argument="goodbye")

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



if __name__ == '__main__':
    unittest.main()