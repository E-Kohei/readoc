import io
import numbers
import numpy as np
import fitz
import cv2
from collections import defaultdict
from functools import lru_cache
from PIL import Image

from readoc import Pattern, document_cache
from readoc.pdf.utils import Rect, Cell, Table


def get_bbox(contour):
    min_x = contour[:,0,0].min()
    max_x = contour[:,0,0].max()
    min_y = contour[:,0,1].min()
    max_y = contour[:,0,1].max()
    return Rect(min_x, min_y, max_x, max_y)


def logarithm_threshold(bbox_whs):
        non_zero_whs = bbox_whs[np.all(bbox_whs > 0, axis=1)]
        bbox_whs_log = np.log2(non_zero_whs)
        widths = bbox_whs_log[:,0].copy()
        heights = bbox_whs_log[:,1].copy()
        widths.sort()
        heights.sort()
        w_diffs = np.diff(widths)
        h_diffs = np.diff(heights)
        w_threshold = np.exp2( widths[w_diffs.argmax()] + w_diffs.max()/2 )
        h_threshold = np.exp2( heights[h_diffs.argmax()] + h_diffs.max()/2 )

        return (w_threshold, h_threshold)


@document_cache
def convert_to_img(document: 'fitz.Document',  page, resolution_rate=1, **kwargs):
    '''Get an image of the document as numpy::array.

    Parameters
    ----------
    document: fitz.Document
        Document to convert to an image.
    
    page: int
        page of ducument to convert to an image.
    
    resolution_rate: int or float
        Resolution rate of the image.
    
    **kwargs: dict, optional
        There is no available extra arguments.
    '''
    resolution_matrix = fitz.Matrix(resolution_rate, resolution_rate)
    pixmap = document[page].get_pixmap(matrix=resolution_matrix)
    img = np.asarray(Image.open(io.BytesIO(pixmap.tobytes())))
    return img
    

@document_cache
def convert_to_binary_img(document: 'fitz.Document', page, binary_threshold="auto", **kwargs):
    '''Get a binary image of the document as numpy::array.

    Parameters
    ----------
    document: fitz.Document
        Document to convert an image.
    
    page: int
        page of the document to convert to a binary image.
    
    binary_threshold: int
        threshold to gray scale color.
    
    **kwargs: dict, optional
        Extra arguments to `resolution_rate`: refer to `PDFPattern.convert_to_img`.
    '''
    img = convert_to_img(document, page=page, **kwargs)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if binary_threshold == "auto":
        _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        _, img_bin = cv2.threshold(img_gray, binary_threshold, 255, cv2.THRESH_BINARY_INV)
    return img_bin


@document_cache
def convert_to_drawings_img(document: 'fitz.Document', page, character_size_threshold="auto", **kwargs):
    '''Get an image which contains possibly only drawings (tables, chars, images, i.e. opposite of characters).

    Parameters
    ----------
    document: fitz.Document
        Document to extract drawings from.
    
    page: int
        page of the document to extract drawings from.
    
    character_size_threshold: "auto" or int
        If int, Drawings with size under this value is regarded as characters (or their components).
        If "auto", this value is calculated by `logarithm_threshold` function.
    
    **kwargs: dict, optional
        Extra arguments
        `resolution_rate`: refer to `PDFPattern.convert_to_img`.
        `binary_threshold`: refer to `PDFPattern.convert_to_binary_img`.
    '''
    img_bin = convert_to_binary_img(document, page=page, **kwargs)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_ext = tuple( cont for i,cont in enumerate(contours) if hierarchy[0][i][3] == -1 )
    bbox_whs = np.array([(bbox[2]-bbox[0], bbox[3]-bbox[1]) for bbox in map(get_bbox, contours_ext)])

    img_cp = img_bin.copy()
    
    if character_size_threshold == "auto":
        w_threshold, _ = logarithm_threshold(bbox_whs)
        h_threshold = np.inf
    else:
        w_threshold = h_threshold = character_size_threshold
    
    for i, cont in enumerate(contours_ext):
        # remove characters (which should be small figures) by filling them black
        if bbox_whs[i,0] < w_threshold and bbox_whs[i,1] < h_threshold:
            cv2.drawContours(img_cp, contours_ext, i, 0, -1)
    
    return img_cp


@document_cache
def convert_to_characters_img(document: 'fitz.Document', page, dilate_kernel=np.ones((6,6), np.uint8), **kwargs):
    '''Get an image which contains possibly only characters.
    
    Parameters
    ----------
    document: fitz.Document
        Document to extract characters.
    
    page: int
        page of the document to extract characters from.
    
    dilate_kernel: numpy::ndarry
        dilate kernel.
    
    **kwargs: dict, optinal
        Extra arguments
        `resolution_rate`: refer to `PDFPattern.convert_to_img`.
        `binary_threshold`: refer to `PDFPattern.convert_to_binary_img`.
        `character_size_threshold`: refer to `Pattern.convert_to_drawings_img`.
    '''
    img_bin = convert_to_binary_img(document, page=page, **kwargs)
    img_drawings = convert_to_drawings_img(document, page=page, **kwargs)
    img_drawings = cv2.dilate(img_drawings, kernel=dilate_kernel)
    img_characters = np.where(img_drawings==255, 0, img_bin)
    return img_characters


@document_cache
def extract_word_boxes(document: 'fitz.Document', page, resolution_rate=1, **kwargs):
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
    word_boxes = document[page].get_text(option="words", sort=True)
    result = [(Rect(wb[:4])*resolution_rate,wb[4]) for wb in word_boxes]
    return result


@document_cache
def extract_text(document: 'fitz.Document', page, clip, **kwargs):
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
    word_boxes = extract_word_boxes(document, page=page, **kwargs)
    text = ""
    for wb in word_boxes:
        if clip.includes(wb[0]):
            text += wb[1]
    return text


@document_cache
def extract_tables(document: 'fitz.Document', page, min_cell_width=10, **kwargs):
    '''Extract tables from the document.

    Parameters
    ----------
    document: fitz.Document
        Document to extract tables from.
    
    page: int
        page of the document to extract tables from.
    
    min_cell_width: int or float
        minimum cell width with which cells are regarded as outlier.
    
    **kwargs: dict, optional
        Extra arguments
        `resolution_rate`: refer to `convert_to_img`.
        `binary_threshold`: refer to `convert_to_binary_img`.
        `character_size_threshold`: refer to `convert_to_drawings_img`.
    '''
    img_tables = convert_to_drawings_img(document, page=page, **kwargs)
    contours, hierarchy = cv2.findContours(img_tables, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tables_raw = defaultdict(list)
    for i, cont in enumerate(contours):
        if hierarchy[0][i][2] == -1 and hierarchy[0][i][3] != -1:    # the most inside cell
            bbox = get_bbox(cont)
            if bbox.width() >= min_cell_width:
                tables_raw[hierarchy[0][i][3]].append(Cell(bbox, ""))
    tables = []
    for cells in tables_raw.values():
        table = Table.from_cells(cells)
        tables.append(table)
    return tables


def _parse_position(position, page_rect, resolution_rate=1):
    page_rect = page_rect * resolution_rate
    actual_position = list(position)
    for i, p in enumerate(position):
        if p == "*" and (i == 0 or i == 1):
            actual_position[i] = -np.inf
        elif p == "*" and (i == 2 or i == 3):
            actual_position[i] = np.inf
        elif 0 <= p < 1 and (i == 0 or i == 1):
            actual_position[i] = position[i] * page_rect[i]
        elif 0 <= p <= 1 and (i == 2 or i == 3):
            actual_position[i] = position[i] * page_rect[i]
    return Rect(actual_position)


def _parse_cell_loc(cell_loc, actual_shape):
    loc = list(cell_loc)
    for i,c in enumerate(loc):
        if c < 0:
            loc[i] = actual_shape[i] + c
    return tuple(loc)


def _parse_cell_range(cell_range, actual_shape):
    start = _parse_cell_loc(cell_range[0], actual_shape)
    end = _parse_cell_loc(cell_range[1], actual_shape)
    cell_coords = []
    for row in range(start[0],end[0]+1):
        for col in range(start[1],end[1]+1):
            cell_coords.append((row,col))
    return cell_coords


def _find_match_table(tables, actual_position, expr_shape):
    matched_table = None
    actual_shape = None
    for table in tables:
        if actual_position.includes(table.get_bbox()) and _does_shape_match(expr_shape, table.shape):
            matched_table = table
            actual_shape = table.shape
            break
    return matched_table, actual_shape


def _does_shape_match(expr_shape, actual_shape):
    for s1,s2 in zip(expr_shape, actual_shape):
        if s1 == "*":
            pass
        elif s1 != s2:
            return False
    return True



class PDFPattern(Pattern):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def process_document(self, document: 'fitz.Document'):
        if not isinstance(document, fitz.Document):
            raise TypeError("The argument `document` must be fitz::Page.")



class TextBoxPattern(PDFPattern):
    def __init__(self, page, position, action, name="extracted_text", expected_text=None, exact_match=False, remove_newlines=True, **kwargs):
        super().__init__(**kwargs)
        self.page = page
        self.position = position
        self.action = action
        self.name = name
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
        extracted_text = extract_text(document, self.page, actual_position)
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



class MatrixTablePattern(PDFPattern):
    def __init__(self, page, position, shape, landmarks, table_headers, additional_info_cells, resolution_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.page = page
        self.position = position
        self.shape = shape
        self.landmarks = landmarks
        self.table_headers = table_headers
        self.additional_info_cells = additional_info_cells
        self.resolution_rate = resolution_rate
        self.other_kwargs = kwargs
    
    def process_document(self, document: 'fitz.Document'):
        tables = extract_tables(document, self.page, resolution_rate=self.resolution_rate, **self.other_kwargs)
        actual_position = _parse_position(self.position, document[self.page].rect, self.resolution_rate)
        check_count = 0
        match_count = 0
        result_value = {}

        # find match table
        check_count += 1
        matched_table, actual_shape = _find_match_table(tables, actual_position, self.shape)
        if matched_table is None:
            return {"check_count": check_count, "match_score": match_count / check_count, "value": None}
        else:
            match_count += 1
        
        # landmarks are used to calculate match score
        for landmark in self.landmarks:
            check_count += 1
            loc = _parse_cell_loc(landmark["loc"], actual_shape)
            matched_cell = matched_table[loc]
            extracted_text = extract_text(document, self.page, matched_cell.bbox,
                                          resolution_rate=self.resolution_rate, **self.other_kwargs)
            if landmark["expected_text"] == extracted_text:
                match_count += 1
        
        # table_headers are used to extract matrix data
        header_dict = defaultdict(list)
        for table_header in self.table_headers:
            loc = _parse_cell_loc(table_header["loc"], actual_shape)
            matched_cell = matched_table[loc]
            extracted_text = extract_text(document, self.page, matched_cell.bbox,
                                          resolution_rate=self.resolution_rate, **self.other_kwargs)
            if "expected_text" in table_header:
                check_count += 1
                if table_header["expected_text"] == extracted_text:
                    match_count += 1
            for header_loc in _parse_cell_range(table_header["scope-range"], actual_shape):
                header_dict[header_loc].append(table_header["name"])
        for loc, header_names in header_dict.items():
            key = tuple(header_names)
            val = extract_text(document, self.page, matched_table[loc].bbox,
                               resolution_rate=self.resolution_rate, **self.other_kwargs)
            result_value[key] = val
        
        # additional_info_cells are used to extract cell contents
        for additional_info_cell in self.additional_info_cells:
            loc = _parse_cell_loc(additional_info_cell["loc"], actual_shape)
            matched_cell = matched_table[loc]
            extracted_text = extract_text(document, self.page, matched_cell.bbox,
                                          resolution_rate=self.resolution_rate, **self.other_kwargs)
            result_value[additional_info_cell["name"]] = extracted_text
        
        return {
            "check_count": check_count,
            "match_score": match_count / check_count,
            "value": result_value
        }



class ListTablePattern(PDFPattern):
    def __init__(self, page, position, shape, grouping, landmarks, table_headers, additional_info_cells, resolution_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.page = page
        self.position = position
        self.shape = shape
        self.grouping = grouping
        self.landmarks = landmarks
        self.table_headers = table_headers
        self.additional_info_cells = additional_info_cells
        self.resolution_rate = resolution_rate
        self.other_kwargs = kwargs
    
    def process_document(self, document: 'fitz.Document'):
        tables = extract_tables(document, self.page, resolution_rate=self.resolution_rate, **self.other_kwargs)
        actual_position = _parse_position(self.position, document[self.page].rect, self.resolution_rate)
        check_count = 0
        match_count = 0
        result_value = {}

        # find match table
        check_count += 1
        matched_table, actual_shape = _find_match_table(tables, actual_position, self.shape)
        if matched_table is None:
            return {"check_count": check_count, "match_score": match_count / check_count, "value": None}
        else:
            match_count += 1
        
        # landmarks are used to calculate match score
        for landmark in self.landmarks:
            check_count += 1
            loc = _parse_cell_loc(landmark["loc"], actual_shape)
            matched_cell = matched_table[loc]
            extracted_text = extract_text(document, self.page, matched_cell.bbox,
                                          resolution_rate=self.resolution_rate, **self.other_kwargs)
            if landmark["expected_text"] == extracted_text:
                match_count += 1
        
        # table_headers are used to extract listed data
        header_dict = defaultdict(list)
        for table_header in self.table_headers:
            loc = _parse_cell_loc(table_header["loc"], actual_shape)
            matched_cell = matched_table[loc]
            extracted_text = extract_text(document, self.page, matched_cell.bbox,
                                          resolution_rate=self.resolution_rate, **self.other_kwargs)
            if "expected_text" in table_header:
                check_count += 1
                if table_header["expected_text"] == extracted_text:
                    match_count += 1
            for header_loc in _parse_cell_range(table_header["scope-range"], actual_shape):
                header_dict[header_loc].append(table_header["name"])
        items = []
        if self.grouping == "row":
            for row in range(actual_shape[0]):
                item = dict([
                    ( tuple(header_dict[row,col]), extract_text(document, self.page, matched_table[row,col].bbox, resolution_rate=self.resolution_rate, **self.other_kwargs) )
                    for col in range(actual_shape[1])
                    if len(header_dict[row,col]) > 0
                ])
                if len(item) > 0:
                    items.append(item)
        elif self.grouping == "col":
            for col in range(actual_shape[1]):
                item = dict([
                    ( tuple(header_dict[row,col]), extract_text(document, self.page, matched_table[row,col].bbox, resolution_rate=self.resolution_rate, **self.other_kwargs) )
                    for row in range(actual_shape[0])
                    if len(header_dict[row,col]) > 0
                ])
                if len(item) > 0:
                    items.append(item)
        result_value["items"] = items
        
        # additional_info_cells are used to extract cell contents
        for additional_info_cell in self.additional_info_cells:
            loc = _parse_cell_loc(additional_info_cell["loc"], actual_shape)
            matched_cell = matched_table[loc]
            extracted_text = extract_text(document, self.page, matched_cell.bbox,
                                          resolution_rate=self.resolution_rate, **self.other_kwargs)
            result_value[additional_info_cell["name"]] = extracted_text
        
        return {
            "check_count": check_count,
            "match_score": match_count / check_count,
            "value": result_value
        }
