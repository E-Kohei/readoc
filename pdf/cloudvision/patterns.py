import fitz
import numpy as np

from google.cloud import vision

from readoc import document_cache
from readoc.pdf.patterns import PDFPattern, _parse_position
from readoc.pdf.utils import Rect



@document_cache
def extract_word_boxes_google_vision_ocr(document: 'fitz.Document', page, resolution_rate=1, **kwargs):
    '''Get list of word boxes of the document.
    
    Parameters
    ----------
    document: fitz.Document
        Document to get the list of word boxes from.
    
    page: int
        page of the document to extract word-boxes from.

    resolution_rate: int or float
        Resolution rate of the image. 
        The result word boxes are scaled by this value to match the size of image processing functions.
    '''
    client = vision.ImageAnnotatorClient()

    resolution_matrix = fitz.Matrix(resolution_rate, resolution_rate)
    pixmap = document[page].get_pixmap(matrix=resolution_matrix)
    content = pixmap.tobytes()
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    text_annotation = response.full_text_annotation

    # TextAnnotation's structure: TextAnnotation -> Page -> Block -> Paragraph -> Word -> Symbol
    width  = document[page].rect.width
    height = document[page].rect.height
    word_boxes = []
    for page in text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    bbox_verts = word.bounding_box.vertices
                    bbox = Rect(bbox_verts[0].x, bbox_verts[0].y,
                                bbox_verts[2].x, bbox_verts[2].y)
                    concatenated_word = ""
                    for symbol in word.symbols:
                        concatenated_word += symbol.text
                    word_boxes.append( (bbox, concatenated_word) )
    return word_boxes


def extract_text_google_vision_ocr(document: 'fitz.Document', page, clip, **kwargs):
    '''Extract text inside the specified clip rectangle in the document.

    Parameters
    ----------
    document: fitz.Document
        Document to extract text from.
    
    page: int
        page of the document to extract text from.

    clip: Rect-like
        Clip rectangle in the document.
    
    **kwargs: dict, optional
        Extra arguments
        `resolution_rate`: refer to `extract_word_boxes`.
    '''
    clip = Rect(clip)
    word_boxes = extract_word_boxes_google_vision_ocr(document, page=page, **kwargs)
    text = ""
    for wb in word_boxes:
        if clip.includes(wb[0]):
            text += wb[1]
    return text



class GoogleVisionOCRPattern(PDFPattern):
    def __init__(self, page, position, action, name="extracted_text", expected_text=None, exact_match=False, remove_newlines=True, **kwargs):
        super().__init__(**kwargs)
        self.page = page
        self.position = position
        self.action = action
        self.name = name
        self.expected_text = expected_text
        self.exact_match = exact_match
        self.remove_newlines = remove_newlines
        if self.action == "check_format":
            if expected_text is None:
                raise TypeError("The argument `expected_text` is required if `action` is 'check_format'")
            self.expected_text = expected_text
        elif self.action == "extract_text":
            pass
        else:
            raise TypeError("The argument `action` must be either 'check_format' or 'extract_text'")
    
    def process_document(self, document: 'fitz.Document'):
        super().process_document(document)
        actual_position = _parse_position(self.position, document[self.page].rect)
        extracted_text = extract_text_google_vision_ocr(document, self.page, clip=actual_position)
        if self.remove_newlines:
            extracted_text = extracted_text.replace('\n', '')
        
        if self.action == "check_format":
            if self.exact_match:
                match_score = 1 if self.expected_text == extracted_text else 0
                return {"check_count": 1, "match_score": match_score, "value":None}
            else:
                check_count = len(self.expected_text)
                match_count = 0
                for word in self.expected_text:
                    if word in extracted_text:
                        match_count += 1
                return {"check_count": check_count, "match_score": match_count/check_count, "value":None}
        
        elif self.action == "extract_text":
            return {
                "check_count": None,
                "match_score": None,
                "value": {
                    self.name: extracted_text
                }
            }

        