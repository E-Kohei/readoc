import io
import numbers
import numpy as np
import fitz
import cv2
from collections import defaultdict
from PIL import Image

from readoc.core import Pattern, document_cache
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



class PDFPattern(Pattern):
    '''Pattern for PDF Document

    This is an abstract class. Override this class to create patterns for PDF document.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    
    @document_cache(share_with_other_patterns=True)
    def convert_to_img(self, document: 'fitz.Document',  page, resolution_rate=1, **kwargs):
        '''Get an image of the document as numpy::array.

        Parameters
        ----------
        document: fitz.Document
          Document to convert to an image and cache storage of the result.
        
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
    

    @document_cache(share_with_other_patterns=True)
    def convert_to_binary_img(self, document: 'fitz.Document', page, binary_threshold=210, **kwargs):
        '''Get a binary image of the document as numpy::array.

        Parameters
        ----------
        document: fitz.Document
          Document to convert an image and cache storage of the result.
        
        page: int
          page of the document to convert to a binary image.
        
        binary_threshold: int
          threshold to gray scale color.
        
        **kwargs: dict, optional
          Extra arguments to `resolution_rate`: refer to `PDFPattern.convert_to_img`.
        '''
        img = self.convert_to_img(document, page=page, **kwargs)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, img_bin = cv2.threshold(img_gray, binary_threshold, 255, cv2.THRESH_BINARY_INV)
        return img_bin
    

    @document_cache(share_with_other_patterns=True)
    def convert_to_drawings_img(self, document: 'fitz.Document', page, character_size_threshold="auto", **kwargs):
        '''Get an image which contains possibly only drawings (tables, chars, images, i.e. opposite of characters).

        Parameters
        ----------
        document: fitz.Document
          Document to extract drawings from and cache storage of the result.
        
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
        img_bin = self.convert_to_binary_img(document, page=page, **kwargs)
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
    

    @document_cache(share_with_other_patterns=True)
    def convert_to_characters_img(self, document: 'fitz.Document', page, dilate_kernel=np.ones((6,6), np.uint8), **kwargs):
        '''Get an image which contains possibly only characters.
        
        Parameters
        ----------
        document: fitz.Document
          Document to extract characters from and cache storage of the result.
        
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
        img_bin = self.convert_to_binary_img(document, page=page, *kwargs)
        img_drawings = self.convert_to_drawings_img(document, page=page, **kwargs)
        img_drawings = cv2.dilate(img_drawings, kernel=dilate_kernel)
        img_characters = np.where(img_drawings==255, 0, img_bin)
        return img_characters
    

    @document_cache(share_with_other_patterns=False)
    def extract_word_boxes(self, document: 'fitz.Document', page, **kwargs):
        '''Get list of word boxes of the document.
        
        Parameters
        ----------
        document: fitz.Document
          Document to get the list of word boxes from.
        
        page: int
          page of the document to extract word-boxes from.
        '''
        word_boxes = document[page].get_text(option="words", sort=True)
        result = [(Rect(wb[:4]),wb[4]) for wb in word_boxes]
        return result
    

    def extract_text(self, document: 'fitz.Document', page, clip):
        '''Extract text inside the specified clip rectangle in the document.

        Parameters
        ----------
        document: fitz.Document
          Document to extract text from.
        
        page: int
          page of the document to extract text from.

        clip: Rect-like
          Clip rectangle in the document.
        '''
        clip = Rect(clip)
        word_boxes = self.extract_word_boxes(document, page=page)
        text = ""
        for wb in word_boxes:
            if clip.includes(wb[0]):
                text += wb[1]
        return text
    

    def calc_position(self, document, page, position, resolution_rate=1):
        page_rect = document[page].rect * resolution_rate
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


    def process_document(self, document: 'fitz.Document'):
        if not isinstance(document, fitz.Document):
            raise TypeError("The argument `document` must be fitz::Page.")



class TextBoxPattern(PDFPattern):
    def __init__(self, page, position, action, target_label="extracted_text", expected_text=None, remove_newlines=True, **kwargs):
        super().__init__(**kwargs)
        self.page = page
        self.position = position
        self.action = action
        self.target_label = target_label
        self.remove_newlines = remove_newlines
        if self.action == "check_format" and expected_text is not None:
            self.expected_text = expected_text
        elif self.action == "extract_text":
            pass
        else:
            raise TypeError("The argument `action` must be either 'check' or 'extract'")
    
    def process_document(self, document: 'fitz.Document'):
        super().process_document(document)
        actual_position = self.calc_position(document, self.page, self.position)
        extracted_text = document[self.page].get_text(clip=actual_position)
        if self.remove_newlines:
            extracted_text = extracted_text.replace('\n', '')
        if self.action == "check_format":
            print(f"t = {extracted_text}")
            if self.expected_text == extracted_text:
                return {"check_count": 1, "match_score": 1, "value":None}
            else:
                return {"check_count": 1, "match_score": 0, "value": None}
        elif self.action == "extract_text":
            return {
                "check_count": None,
                "match_score": None,
                "value": {
                    self.target_label: extracted_text
                }
            }



class MatrixTablePattern(PDFPattern):
    def __init__(self, page, position, shape, landmarks, table_headers, additional_cells, **kwargs):
        super().__init__(**kwargs)
        self.page = page
        self.position = position
        self.shape = shape
        self.landmarks = landmarks
        self.table_headers = table_headers
        self.additional_cells = additional_cells
        self._matched_table = None
        self._actual_shape = None
    
    @document_cache(share_with_other_patterns=True)
    def extract_tables(self, document: 'fitz.Document', page, **kwargs):
        img_tables = self.convert_to_drawings_img(document, page=page, **kwargs)
        contours, hierarchy = cv2.findContours(img_tables, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tables_raw = defaultdict(list)
        for i, cont in enumerate(contours):
            if hierarchy[0][i][2] == -1 and hierarchy[0][i][3] != -1:    # the most inside cell
                bbox = get_bbox(cont)
                tables_raw[hierarchy[0][i][3]].append(Cell(bbox, ""))
        tables = []
        for cells in tables_raw.values():
            table = Table(cells)
            tables.append(table)
        return tables
    
    def find_match_table(self, document, page):
        tables = self.extract_tables(document, page)
        actual_position = self.calc_position(document, self.page, self.position)
        for table in tables:
            if actual_position.includes(table.get_bbox()) and self.does_shape_match(table.shape):
                self._matched_table = table
                self._actual_shape = table.shape
                break
        return self._matched_table
        
    def parse_range(self, cell_range):
        if len(cell_range) == 1:
            start = end = cell_range
        else:
            start, end = cell_range
        cell_coords = []
        for i,c in start:
            if c < 0:
                start[i] = self._actual_shape[i] - c
        for i,c in end:
            if c < 0:
                end[i] = self._actual_shape[i] - c
        for row in range(start[0],end[0]+1):
            for col in range(start[1],end[1]+1):
                cell_coords.append((row,col))
        return cell_coords
    
    def does_shape_match(self, shape):
        for s1,s2 in zip(self.shape, shape):
            if s1 == "*":
                pass
            elif s1 != s2:
                return False
        return True
    
    def process_document(self, document: 'fitz.Document'):
        word_boxes = self.extract_word_boxes(document, self.page)
        tables = self.extract_tables(document, self.page)
        check_count = 0
        match_count = 0
        result_value = {}

        # find match table
        check_count += 1
        matched_table = self.find_match_table(document, self.page)
        if matched_table is None:
            return {"check_count": check_count, "match_score": match_count / check_count, "value": None}
        else:
            match_count += 1
        
        # landmarks are used to calculate match score
        for landmark in self.landmarks:
            check_count += 1
            locs = self.parse_range(landmark["loc"])
            matched_cell = matched_table[locs[0]]
            extracted_text = self.extract_text(document, self.page, matched_cell.bbox)
            if landmark["expected_text"] == extracted_text:
                match_count += 1
        
        # table_headers are used to extract matrix data
        header_dict = defaultdict(list)
        for table_header in self.table_headers:
            locs = self.parse_range(table_header["loc"])
            matched_cell = matched_table[locs[0]]
            extracted_text = self.extract_text(document, self.page, matched_cell.bbox)
            if "expected_text" in table_header:
                check_count += 1
                if table_header["expected_text"] == extracted_text:
                    match_count += 1
            for loc in self.parse_range(table_header["scope-range"]):
                header_dict[loc].append(table_header["name"])
        for loc, header_names in header_dict.items():
            key = tuple(header_names)
            val = self.extract_text(document, self.page, matched_table[loc].bbox)
            result_value[key] = val
        
        # additional_cells are used to extract cell contents
        for additional_cell in self.additional_cells:
            locs = self.parse_range(additional_cell["loc"])
            matched_cell = matched_table[locs[0]]
            extracted_text = self.extract_text(document, self.page, matched_cell.bbox)
            result_value[additional_cell["name"]] = extracted_text
        
        return {
            "check_count": check_count,
            "match_score": match_count / check_count,
            "value": result_value
        }



class ListTablePattern(PDFPattern):
    def __init__(self, page, position, shape, grouping, landmarks, table_headers, additional_cells, **kwargs):
        super().__init__(**kwargs)
        self.page = page
        self.position = position
        self.shape = shape
        self.grouping = grouping
        self.landmarks = landmarks
        self.table_headers = table_headers
        self.additional_cells = additional_cells
        self._matched_table = None
        self._actual_shape = None

    @document_cache(share_with_other_patterns=True)
    def extract_tables(self, document: fitz.Document, page, **kwargs):
        img_tables = self.convert_to_drawings_img(document, page=page, **kwargs)
        contours, hierarchy = cv2.findContours(img_tables, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        tables_raw = defaultdict(list)
        for i, cont in enumerate(contours):
            if hierarchy[0][i][2] == -1 and hierarchy[0][i][3] != -1:    # the most inside cell
                bbox = get_bbox(cont)
                tables_raw[hierarchy[0][i][3]].append(Cell(bbox, ""))
        tables = []
        for cells in tables_raw.values():
            table = Table(cells)
            tables.append(table)
        return tables
    
    def find_match_table(self, document):
        tables = self.extract_tables(document, self.page)
        actual_position = self.calc_position(document, self.page, self.position)
        for table in tables:
            if actual_position.includes(table.get_bbox()) and self.does_shape_match(table.shape):
                self._matched_table = table
                self._actual_shape = table.shape
                break
        return self._matched_table
    
    def parse_range(self, cell_range):
        if len(cell_range) == 1:
            start = end = cell_range
        else:
            start, end = cell_range
        cell_coords = []
        for i,c in start:
            if c < 0:
                start[i] = self._actual_shape[i] - c
        for i,c in end:
            if c < 0:
                end[i] = self._actual_shape[i] - c
        for row in range(start[0],end[0]+1):
            for col in range(start[1],end[1]+1):
                cell_coords.append((row,col))
        return cell_coords
    
    def does_shape_match(self, shape):
        for s1,s2 in zip(self.shape, shape):
            if s1 == "*":
                pass
            elif s1 != s2:
                return False
        return True
    
    def process_document(self, document: 'fitz.Document'):
        word_boxes = self.extract_word_boxes(document)
        tables = self.extract_tables(document)
        check_count = 0
        match_count = 0
        result_value = {}

        # find match table
        check_count += 1
        matched_table = self.find_match_table(document)
        if matched_table is None:
            return {"check_count": check_count, "match_score": match_count / check_count, "value": None}
        else:
            match_count += 1
        
        # landmarks are used to calculate match score
        for landmark in self.landmarks:
            check_count += 1
            locs = self.parse_range(landmark["loc"])
            matched_cell = matched_table[locs[0]]
            extracted_text = self.extract_text(document, self.page, matched_cell.bbox)
            if landmark["expected_text"] == extracted_text:
                match_count += 1
        
        # table_headers are used to extract listed data
        header_dict = defaultdict(list)
        for table_header in self.table_headers:
            locs = self.parse_range(table_header["loc"])
            matched_cell = matched_table[locs[0]]
            extracted_text = self.extract_text(document, self.page, matched_cell.bbox)
            if "expected_text" in table_header:
                check_count += 1
                if table_header["expected_text"] == extracted_text:
                    match_count += 1
            for loc in self.parse_range(table_header["scope-range"]):
                header_dict[loc].append(table_header["name"])
        items = []
        if self.grouping == "row":
            for row in range(self._actual_shape[0]):
                item = dict([
                    ( tuple(header_dict[row,col]), self.extract_text(document, self.page, matched_table[row,col].bbox) )
                    for col in range(self._actual_shape[1])
                    if len(header_dict[row,col]) > 0
                ])
                items.append(item)
        elif self.grouping == "col":
            for col in range(self._actual_shape[1]):
                item = dict([
                    ( tuple(header_dict[row,col]), self.extract_text(document, self.page, matched_table[row,col].bbox) )
                    for row in range(self._actual_shape[0])
                    if len(header_dict[row,col]) > 0
                ])
                items.append(item)
        result_value["items"] = items
        
        # additional_cells are used to extract cell contents
        for additional_cell in self.additional_cells:
            locs = self.parse_range(additional_cell["loc"])
            matched_cell = matched_table[locs[0]]
            extracted_text = self.extract_text(document, self.page, matched_cell.bbox)
            result_value[additional_cell["name"]] = extracted_text
        
        return {
            "check_count": check_count,
            "match_score": match_count / check_count,
            "value": result_value
        }




if __name__ == '__main__':
    def create_test_pdf():
        doc = fitz.open()
        page = doc.new_page()
        text1 = "Test PDF"
        text2 = "This is \n an English text"
        text3 = "日本語\n  文字列"
        p1 = fitz.Point(25, 25)
        p2 = fitz.Point(page.rect.width-100, 25)
        p3 = fitz.Point(page.rect.width/2, page.rect.height/2)
        
        shape = page.new_shape()
        shape.draw_circle(p1, 1)
        shape.draw_circle(p2, 1)
        shape.draw_circle(p3, 1)

        shape.insert_text(p1, text1)
        shape.insert_text(p2, text2)
        shape.insert_text(p3, text3)

        shape.commit()
        return doc