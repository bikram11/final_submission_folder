class GridSequencer(object):
    def __init__(self, row):
        self._data = row

    @property
    def img_number(self):
        return (self._data[0]) 

    @ property
    def grid_matrix(self):
        grid_row=[]
        for item in self._data[1:]:
            grid_row.append(float(item))
        return grid_row