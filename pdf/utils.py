import numbers
import re
import numpy as np

from copy import deepcopy
from enum import Enum
from functools import lru_cache


class Rect(tuple):
    def __new__(cls, *args):
        try:
            if len(args) == 4 and Rect.is_all_number(args):
                return tuple.__new__(cls, args)
            elif len(args) == 2 and len(args[0]) == len(args[1]) == 2 and \
                 Rect.is_all_number(args[0]) and Rect.is_all_number(args[1]):
                return tuple.__new__(cls, (*args[0], *args[1]))
            elif len(args) == 1 and len(args[0]) == 4 and Rect.is_all_number(args[0]):
                return tuple.__new__(cls, args[0])
        except:
            raise TypeError("Invalid args for Rect")
    
    @staticmethod
    def is_all_number(iterable):
        for x in iterable:
            if not isinstance(x, numbers.Number):
                return False
        return True
    
    def __add__(self, other):
        return Rect(tuple(self[i]+other[i] for i in range(4)))
    
    def __mul__(self, value):
        return Rect(tuple(x*value for x in self))
    
    def __floordiv__(self, value):
        return Rect(tuple(x//value for x in self))
    
    def width(self):
        return self[2] - self[0]
    
    def height(self):
        return self[3] - self[1]
    
    def includes(self, other):
        if self[0] <= other[0] <= other[2] <= self[2] and \
           self[1] <= other[1] <= other[3] <= self[3]:
            return True
        else:
            return False



class Cell():
    def __init__(self, bbox, content, row_group=None, col_group=None):
        if not isinstance(content, str):
            raise TypeError("Cell content must be str")
        self.bbox = Rect(bbox)
        self.content = content
        if row_group is None:
            self.row_group = set()
        if col_group is None:
            self.col_group = set()
    
    def __getitem__(self, key):
        return self.__getattribute__(key)
    
    def __setitem__(self, key, value):
        self.__setattr__(key, value)
    
    def __repr__(self):
        return f"{self.bbox}: {self.content}"
    
    def __eq__(self, other):
        if isinstance(other, Cell):
            return NotImplemented
        return self.bbox == other.bbox and self.content == other.content and \
               self.row_group == other.row_group and self.col_group == other.col_group
    
    def copy(self):
        bbox = Rect(self.bbox)
        content = self.content
        row_group = self.row_group.copy()
        col_group = self.col_group.copy()
        return Cell(bbox, content, row_group, col_group)

    def width(self):
        return self.bbox.width()
    
    def height(self):
        return self.bbox.height()
    
    def includes(self, other: Rect):
        return self.bbox.includes(other.bbox)


class CellRelation(Enum):
    NONE = 0
    SAME_CELL = 1
    SAME_ROW = 2
    SAME_COL = 3
    PARTIALLY_SAME_ROW = 4
    PARTIALLY_SAME_COL = 5
    INSIDE_CELL = 6

    @staticmethod
    def infer_cell_relation(cell1, cell2, threshold=0.9):
        sx1, sy1, ex1, ey1 = cell1["bbox"]
        sx2, sy2, ex2, ey2 = cell2["bbox"]
        w1, h1 = ex1 - sx1, ey1 - sy1
        w2, h2 = ex2 - sx2, ey2 - sy2
        overlap_x = min(w1, w2, ex1 - sx2, ex2 - sx1)
        overlap_y = min(h1, h2, ey1 - sy2, ey2 - sy1)
        
        row_relation = "none"
        col_relation = "none"
        if overlap_x > min(w1,w2)*threshold:
            # the two cell overlaps vertically
            if overlap_x > w1*threshold and overlap_x > w2*threshold:
                col_relation =  "same"
            else:
                col_relation = "partial"
        if overlap_y > min(h1,h2)*threshold:
            # the two cell overlaps horizontally
            if overlap_y > h1*threshold and overlap_y > h2*threshold:
                row_relation = "same"
            else:
                row_relation = "partial"
        
        if row_relation == "same" and col_relation == "same":
            return CellRelation.SAME_CELL
        elif row_relation != "none" and col_relation != "none":
            return CellRelation.INSIDE_CELL
        elif row_relation == "same":
            return CellRelation.SAME_ROW
        elif row_relation == "partial":
            return CellRelation.PARTIALLY_SAME_ROW
        elif col_relation == "same":
            return CellRelation.SAME_COL
        elif col_relation == "partial":
            return CellRelation.PARTIALLY_SAME_COL
        else:
            return CellRelation.NONE  


class Table():
    def __init__(self, shape):
        self.cells = None
        self.cell_matrix = None
        self.shape = None
        if shape is not None:
            nan_cell = Cell(bbox=(np.nan, np.nan, np.nan, np.nan), content="")
            self.cells = []
            self.cell_matrix = np.full(shape, nan_cell.copy(), dtype=object)
            for r,c in np.ndindex(shape):
                self.cells.append(self.cell_matrix[r,c])
            self.shape = shape

    @classmethod
    def from_cells(cls, cells):
        tbl = cls(None)
        tbl.load_cells(cells)
        return tbl
    
    def __getitem__(self, idx):
        return self.cell_matrix[idx]
        
    @staticmethod
    def _group_cells(cells):
        row_count = col_count = 0
        _cells = deepcopy(cells)

        # group cells by row
        processed_cells = []
        _cells = sorted(_cells, key=lambda cell:cell.height())
        for cur_cell in _cells:
            for p_cell in processed_cells:
                relation = CellRelation.infer_cell_relation(cur_cell, p_cell)
                if relation in (CellRelation.SAME_ROW, CellRelation.PARTIALLY_SAME_ROW):
                    cur_cell.row_group.update(p_cell.row_group)
            if len(cur_cell.row_group) == 0:
                cur_cell.row_group.add(row_count)
                row_count += 1
            processed_cells.append(cur_cell)
        
        # group cells by column
        processed_cells.clear()
        _cells = sorted(_cells, key=lambda cell: cell.width())
        for cur_cell in _cells:
            for p_cell in processed_cells:
                relation = CellRelation.infer_cell_relation(cur_cell, p_cell)
                if relation in (CellRelation.SAME_COL, CellRelation.PARTIALLY_SAME_COL):
                    cur_cell.col_group.update(p_cell.col_group)
            if len(cur_cell.col_group) == 0:
                cur_cell.col_group.add(col_count)
                col_count += 1
            processed_cells.append(cur_cell)
        
        return _cells
    
    @staticmethod
    def _sort_cellgroup_index(cells):
        row_group_representatives = []
        col_group_representatives = []
        for cell in cells:
            a_row = next(iter(cell.row_group))
            a_col = next(iter(cell.col_group))
            if len(cell.row_group) == 1 and \
               a_row not in map(lambda repres: repres["row"], row_group_representatives):
                row_group_representatives.append( {"bbox":cell.bbox, "row":a_row} )
            if len(cell.col_group) == 1 and \
               a_col not in map(lambda repres: repres["col"], col_group_representatives):
                col_group_representatives.append( {"bbox":cell.bbox, "col":a_col} )
        
        row_group_representatives.sort(key=lambda repres: repres["bbox"][1])
        col_group_representatives.sort(key=lambda repres: repres["bbox"][0])
        row_index_map = dict( [(repres["row"], idx) for idx, repres in enumerate(row_group_representatives)] )
        col_index_map = dict( [(repres["col"], idx) for idx, repres in enumerate(col_group_representatives)] )

        for cell in cells:
            cell.row_group = { row_index_map[orig_idx] for orig_idx in cell.row_group }
            cell.col_group = { col_index_map[orig_idx] for orig_idx in cell.col_group }
    
    def load_cells(self, cells):
        self.cells = Table._group_cells(cells)
        Table._sort_cellgroup_index(self.cells)

        num_row = max([max(cell.row_group) for cell in self.cells]) + 1
        num_col = max([max(cell.col_group) for cell in self.cells]) + 1
        nan_cell = Cell(bbox=(np.nan, np.nan, np.nan, np.nan), content="")
        self.cell_matrix = np.full((num_row,num_col), nan_cell, dtype=object)

        for cell in self.cells:
            for row in cell.row_group:
                for col in cell.col_group:
                    self.cell_matrix[row,col] = cell
        
        #complement cells
        x1s, y1s, x2s, y2s = [], [], [], []
        for row in range(num_row):
            y1s.append( np.nanmean([cell.bbox[1] for cell in self.cell_matrix[row,:]
                                    if len(cell.row_group) == 1]) )
            y2s.append( np.nanmean([cell.bbox[3] for cell in self.cell_matrix[row,:]
                                    if len(cell.row_group) == 1]) )
        for col in range(num_col):
            x1s.append( np.nanmean([cell.bbox[0] for cell in self.cell_matrix[:,col]
                                    if len(cell.col_group) == 1]) )
            x2s.append( np.nanmean([cell.bbox[2] for cell in self.cell_matrix[:,col]
                                    if len(cell.col_group) == 1]) )
        for row in range(num_row):
            for col in range(num_col):
                if self.cell_matrix[row,col] == nan_cell:
                    self.cell_matrix[row,col] = Cell(bbox=(x1s[col],y1s[row],x2s[col],y2s[row]), content="")
        self.shape = (num_row, num_col)
        return (num_row, num_col)
    
    def is_row_merged_cell(self, i, j):
        return True if len(self[i,j].row_group) >= 2 else False
    
    def is_col_merged_cell(self, i, j):
        return True if len(self[i,j].col_group) >= 2 else False
    
    def get_bbox(self):
        x1 = np.nanmean([cell.bbox[0] for cell in self.cell_matrix[:,0]])
        y1 = np.nanmean([cell.bbox[1] for cell in self.cell_matrix[0,:]])
        x2 = np.nanmean([cell.bbox[2] for cell in self.cell_matrix[:,-1]])
        y2 = np.nanmean([cell.bbox[3] for cell in self.cell_matrix[-1,:]])
        return Rect(x1,y1,x2,y2)
    
    def get_width(self):
        bbox = self.get_bbox()
        return bbox[2] - bbox[0]
    
    def get_height(self):
        bbox = self.get_bbox()
        return bbox[3] - bbox[1]
    
    def get_width_of_col(self, col):
        x1 = np.nanmean([cell.bbox[0] for cell in self.cell_matrix[:,col] if len(cell.col_group) == 1])
        x2 = np.nanmean([cell.bbox[2] for cell in self.cell_matrix[:,col] if len(cell.col_group) == 1])
        return x2 - x1
    
    def get_height_of_row(self, row):
        y1 = np.nanmean([cell.bbox[1] for cell in self.cell_matrix[row,:] if len(cell.row_group) == 1])
        y2 = np.nanmean([cell.bbox[3] for cell in self.cell_matrix[row,:] if len(cell.row_group) == 1])
        return y2 - y1



class RegexDict(dict):
    '''Regex dictionary.
    
    This dictionary can take a regex pattern as a search key.
    '''
    def __getitem__(self, key):
        if isinstance(key, re.Pattern):
            pattern = key
            r = [val for raw_key,val in self.items() if RegexDict._key_search(pattern, raw_key)]
            if len(r) > 0:
                return r
            else:
                raise KeyError(key)
        else:
            return super().__getitem__(key)
    
    def rget(self, *keys):
        patterns = [RegexDict._key_compile(key) for key in keys]
        r = [val for raw_key,val in self.items() 
             if all([RegexDict._key_search(p, raw_key) for p in patterns])]
        if len(r) > 0:
            return r
        else:
            raise KeyError(keys)
    
    def __contains__(self, key):
        if isinstance(key, re.Pattern):
            pattern = key
            m = [RegexDict._key_search(pattern, raw_key) for raw_key in self.keys()]
            return True if any(m) else False
        else:
            return super().__contains__(key)
    
    def is_rin(self, *keys):
        patterns = [RegexDict._key_compile(key) for key in keys]
        m = [all(RegexDict._key_search(p, raw_key) for p in patterns) 
             for raw_key in self.keys()]
        return True if any(m) else False
    
    @staticmethod
    @lru_cache(maxsize=128)
    def _key_compile(key):
        return re.compile(key)
    
    @staticmethod
    def _key_search(pattern, raw_key):
        if isinstance(raw_key, str):
            return re.search(pattern, raw_key)
        elif isinstance(raw_key, tuple):
            m = [pattern.search(k) for k in raw_key]
            return m if any(m) else None

    
