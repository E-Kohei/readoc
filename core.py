import json

PATTERN_CLS={}

class PatternMetaClass(type):
    '''Metaclass of Pattern
    
    This metaclass saves the defined class in `PATTERN_CLS` dictionary.
    '''
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
            pattern_data = json.load(f)
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

    