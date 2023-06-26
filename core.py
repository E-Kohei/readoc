from copy import deepcopy
from inspect import signature
from collections.abc import Hashable
from functools import wraps



PATTERN_CLS={}

# class DocumentData():
#     def __init__(self, document, **kwargs):
#         self.document = document
#         self._cache_data = {}
    
#     def get_document(self):
#         return self.document


class PatternMetaClass(type):
    def __new__(meta, name, bases, attributes):
        # Register new Pattern class to PATTERN_CLS dictionary
        cls = type.__new__(meta, name, bases, attributes)
        PATTERN_CLS[name] = cls
        return cls


class Pattern(metaclass=PatternMetaClass):
    def __init__(self, **kwargs):
        if "score_weight" in kwargs:
            self.score_weight = kwargs["score_weight"]
        else:
            self.score_weight = 1
    
    def process_document(self, document: any):
        pass


class CompositePattern(Pattern):
    def __init__(self, pattern_data):
        self.pattern_data = pattern_data
    
    @classmethod
    def open(cls, pattern_file):
        with open(pattern_file, 'r') as f:
            pattern_data = f.read()
            return cls(pattern_data)
    
    def process_document(self, document: any):
        check_count = 0
        total_score = 0
        total_weight = 0
        value = {}
        for pattern in self.pattern_data["patterns"]:
            cls_name = pattern.pop("class")
            cls = PATTERN_CLS[cls_name]
            pattern = cls(**pattern)
            result = pattern.process_document(document)
            if result["check_count"] is not None:
                check_count += result["check_count"]
            if result["match_score"] is not None:
                total_weight += pattern.score_weight
                total_score += result["match_score"] * pattern.score_weight
            if result["value"] is not None:
                value.update(result["value"])
        return {
            "check_count": check_count,
            "match_score": total_score / total_weight,
            "value": value
        }


class DocumentCache():
    def __new__(cls):
        # Create singleton instance
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, "_cache_data"):
            self._cache_data = {}
    
    @classmethod
    def get_instance(cls):
        return cls()
    
    def __contains__(self, key: Hashable) -> bool:
        return key in self._cache_data
    
    def __len__(self):
        return len(self._cache_data)
    
    def get(self, key: Hashable) -> any:
        return deepcopy(self._cache_data[key])
    
    def set(self, key: Hashable, value: any):
        self._cache_data[key] = deepcopy(value)

    def clear(self):
        self._cache_data.clear()

    def info(self):
        s = cache_length = f"length: {len(self._cache_data)}"
        s += '\n' + '-' * len(cache_length) + '\n'
        for k,v in self._cache_data.items():
            s += f"{k} -> {v}\n"
        return s



def document_cache(share_with_other_patterns=False):
    '''Cache the processing result of Pattern object's method.

    Parameters
    ----------
    share_with_other_patterns : boolean
      If True, cache key does not contain the class (subclass of Pattern) name
      which have the decorated method, i.e the cache data can be shared with other classes.
    '''
    def decorating_function(f):
        @wraps(f)
        def wrapper(obj: 'Pattern', document: any, *args, **kwargs):
            ba = signature(f).bind(obj, document, *args, **kwargs)
            ba.apply_defaults()
            ba.arguments.pop('self')          # key does not contain self object
            sigkey = sorted( convert_hashable(ba.arguments) )
            if share_with_other_patterns:
                key = tuple((f.__name__, *sigkey))
            else:
                key = tuple((f"{obj.__class__.__name__}.{f.__name__}", *sigkey))
            document_cache = DocumentCache.get_instance()
            if key in document_cache:
                return document_cache.get(key)
            else:
                result = f(obj, document, *args, **kwargs)
                document_cache.set(key, result)
                return result
        return wrapper
    
    return decorating_function


def convert_hashable(obj):
    if isinstance(obj, Hashable):
        return obj
    if isinstance(obj, list):
        hashable_seq = []
        for item in obj:
            hashable_seq.append(convert_hashable(item))
        return tuple(hashable_seq)
    if isinstance(obj, dict):
        hashable_map = {}
        for k,v in obj.items():
            hashable_map[k] = convert_hashable(v)
        return tuple(hashable_map.items())


# class WordBoxDetector:
#     def __init__(self):
#         pass
    
#     def listup(self, documentdata):
#         NotImplemented
    
#     def detect(self, documentdata, bbox):
#         NotImplemented

# class OCRWordBoxDetector(WordBoxDetector):
#     def __init__(self, resolution_rate=5, lang="jpn"):
#         self.resolution_rate = resolution_rate
#         self.lang = lang
#         self.tool = pyocr.get_available_tools()[0]
#         self.builder = pyocr.builders.WordBoxBuilder(tesseract_layout=6)
    
#     def listup(self, documentdata: DocumentData):
#         img_chars = documentdata.get_characters_img(resolution_rate=self.resolution_rate)
#         img_chars_pil = Image.fromarray(img_chars)
#         word_boxes = self.tool.image_to_string(img_chars_pil, lang=self.lang, builder=self.builder)
#         result = [(Rect(wb.position)//self.resolution_rate, wb.content) for wb in word_boxes]
#         return result
    
#     def detect(self, documentdata: DocumentData, bbox: Rect, partial_ocr=True):
#         bbox = Rect(bbox)
#         img_chars = documentdata.get_characters_img(resolution_rate=self.resolution_rate)
#         img_chars_pil = Image.fromarray(img_chars)
#         if partial_ocr:
#             word_boxes = self.tool.image_to_string(img_chars_pil.crop(bbox*self.resolution_rate),
#                                                    lang=self.lang,
#                                                    builder=self.builder)
#             text = "".join([wb.content for wb in word_boxes])
#         else:
#             word_boxes = self.tool.image_to_string(img_chars_pil,
#                                                    lang=self.lang,
#                                                    builder=self.builder)
#             text = "".join([wb.content for wb in word_boxes
#                             if Rect(wb.position).includes(bbox*self.resolution_rate)])
#         return text



if __name__ == '__main__':
    mock_document = {
        "title": "Hello, world!",
        "number": 100,
        "text1": "Welcome to Readoc!",
        "text2": "Readoc supports extracting (semi-)structured data from unstructured data.",
        "rawdata": "foo:12,bar:50,baz:100"
    }
    
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
                rawdata = self.get_rawdata(document, additional_vain_argument="foo")
                value.update(rawdata)
            return {
                "check_count": check_count,
                "match_score": match_count,
                "value": value
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
    cp = CompositePattern(pattern_data)
    result = cp.process_document(mock_document)
    print("debug: process_document")
    print(result)
    assert(result["check_count"] == 3)
    assert(result["match_score"] == 102.0)
    assert(result["value"]["foo"] == "12")
    print()

    lst = [1, 2, 3, [10,20], {"foo": {"bar": 100}}]
    hashable_lst = convert_hashable(lst)
    assert(  hashable_lst == ( 1, 2, 3, (10, 20), (('foo', (('bar', 100),)),) )  )

    dp = DictPattern()
    rd = dp.get_rawdata(mock_document, additional_vain_argument="foo")
    key = ('get_rawdata', ('document', (('title', 'Hello, world!'), ('number', 100), ('text1', 'Welcome to Readoc!'), ('text2', 'Readoc supports extracting (semi-)structured data from unstructured data.'), ('rawdata', 'foo:12,bar:50,baz:100'))), ('kwargs', (('additional_vain_argument', 'foo'),)))
    print(PATTERN_CLS)
    