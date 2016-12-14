"""
Maze editor and visualizer.

Created by Petr Viktorin and Miro Hročnok, for the MI-PYT_ course on FIT ČVUT.

| Copyright © 2016 Red Hat, Inc.
| Copyright © 2016 Petr Viktorin
| Copyright © 2016 Miro Hrončok

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Graphics from OpenGameArt.org_ by Kenney_.

.. _MI-PYT: https://github.com/cvut/MI-PYT
.. _OpenGameArt.org: http://opengameart.org/
.. _Kenney: http://opengameart.org/users/kenney
"""
import os

import numpy
from docutils.core import publish_parts
from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg, uic
from bresenham import bresenham

from . import generator
from . import solver
from . import liner

CELL_SIZE = 32
CELL_SIZE_MAX = 128
CELL_SIZE_MIN = 8
ROWS = 31
COLUMNS = 47

KIND_ROLE = QtCore.Qt.UserRole

MAZE_T = numpy.int8
NP_FILES = 'NumPy files (*.csv *.csv.gz)'


def get_filename(name):
    return os.path.join(os.path.dirname(__file__), name)


def get_line_pic(i):
    if 0 < i < 16:
        return QtSvg.QSvgRenderer(get_filename('pics/lines/{}.svg'.format(i)))
    return None


def get_arrow_pic(d):
    return QtSvg.QSvgRenderer(get_filename('pics/arrows/{}.svg'.format(d)))


SVG_GRASS = QtSvg.QSvgRenderer(get_filename('pics/grass.svg'))
SVG_LINES = [get_line_pic(i) for i in range(16)]
SVG_ARROWS = {
    b'>': get_arrow_pic('right'),
    b'<': get_arrow_pic('left'),
    b'^': get_arrow_pic('up'),
    b'v': get_arrow_pic('down'),
}


class GridWidget(QtWidgets.QWidget):
    def __init__(self, array):
        super().__init__()
        self.lines = None
        self._cell_size = CELL_SIZE
        self.array = array
        self.selected_tile_kind = 0
        self.pics = {}

        # drag_start is either None, or (row, col) from where the next line
        # segment drawn by mouse should start
        self.drag_start = None
        self.drag_button = 0

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, val):
        self._array = val
        self._resize()

    @property
    def cell_size(self):
        return self._cell_size

    @cell_size.setter
    def cell_size(self, val):
        self._cell_size = sorted((CELL_SIZE_MIN, val, CELL_SIZE_MAX))[1]
        self._resize()

    def _resize(self):
        size = self.matrix_to_widget_coords(*self.array.shape)
        self.setMinimumSize(*size)
        self.resize(*size)
        self._update()

    def _update(self):
        amaze = solver.analyze(self.array)
        self.directions = amaze.directions
        self.update()
        self.lines = liner.add_lines(amaze.lines, shape=self.array.shape)
        return amaze

    def widget_to_matrix_coords(self, x, y):
        """Given pixel ccordinates, return coordinates of corresponding cell

        Returns (row, column)
        """
        return y // self.cell_size, x // self.cell_size

    def matrix_to_widget_coords(self, row, column):
        """Given cell ccordinates, return corresponding pixel coords

        Returns (x, y)
        """
        return column * self.cell_size, row * self.cell_size

    def paintEvent(self, event):
        rect = event.rect()
        row_min, col_min = self.widget_to_matrix_coords(rect.left(),
                                                        rect.top())
        row_min = max(row_min, 0)
        col_min = max(col_min, 0)
        row_max, col_max = self.widget_to_matrix_coords(rect.right(),
                                                        rect.bottom())
        row_max = min(row_max + 1, self.array.shape[0])
        col_max = min(col_max + 1, self.array.shape[1])
        painter = QtGui.QPainter(self)
        for row in range(row_min, row_max):
            for column in range(col_min, col_max):
                kind = self.array[row, column]
                x, y = self.matrix_to_widget_coords(row, column)
                rect = QtCore.QRectF(x, y, self.cell_size, self.cell_size)
                painter.fillRect(rect,
                                 QtGui.QBrush(QtGui.QColor(255, 255, 255)))
                SVG_GRASS.render(painter, rect)
                if self.lines is not None:
                    line = self.lines[row, column]
                    if line:
                        SVG_LINES[line].render(painter, rect)
                        arrow = self.directions[row, column]
                        if arrow in SVG_ARROWS:
                            SVG_ARROWS[arrow].render(painter, rect)
                if kind != 0:
                    self.pics[kind].render(painter, rect)

    def wheelEvent(self, event):
        if event.modifiers() == QtCore.Qt.ControlModifier:
            degrees = event.angleDelta().y() / 8
            self.cell_size += round(self.cell_size*degrees/100)
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if not self.drag_start or not event.buttons():
            return
        self.drag_to(event.x(), event.y(), self.drag_button)

    def mousePressEvent(self, event):
        if self.drag_start:
            return
        self.drag_button = event.button()
        self.drag_to(event.x(), event.y(), self.drag_button)

    def mouseReleaseEvent(self, event):
        if event.button() & self.drag_button:
            self.drag_to(event.x(), event.y(), self.drag_button)
            self.drag_start = None

    def drag_to(self, end_x, end_y, button):
        end_row, end_column = self.widget_to_matrix_coords(end_x, end_y)
        if self.drag_start:
            start_row, start_column = self.drag_start
            if button & QtCore.Qt.RightButton:
                kind = 0
            else:
                kind = self.selected_tile_kind
            array = self.array
            changed = False
            for row, column in bresenham(start_row, start_column,
                                         end_row, end_column):
                if 0 <= column < array.shape[1] and 0 <= row < array.shape[0]:
                    if array[row, column] != kind:
                        array[row, column] = kind
                        changed = True
            if changed:
                self._update()
        self.drag_start = end_row, end_column


class Gui(object):
    def __init__(self):
        self.app = QtWidgets.QApplication([])

        self.win = win = QtWidgets.QMainWindow()

        self.array = generator.maze(ROWS, COLUMNS)

        self.last_dir = ''
        self.filename = 'untitled.csv.gz'
        self.path = None
        self.new_dialog = None

        with open(get_filename('ui/mainwindow.ui')) as f:
            uic.loadUi(f, win)

        self._update_title()

        icon = QtGui.QIcon(get_filename('pics/dude5.svg'))
        win.setWindowIcon(icon)

        self.scroll_area = self.win.findChild(QtWidgets.QScrollArea, 'scrollArea')
        self.grid = grid = GridWidget(self.array)
        self.scroll_area.setWidget(grid)

        self.palette = palette = self.win.findChild(QtWidgets.QListWidget, 'palette')
        self._add_item('grass', 'Grass', 0, SVG_GRASS)
        self._add_item('wall', 'Wall', -1)
        self._add_item('wall2', 'Unbreakable wall', -2)
        self._add_item('castle', 'Castle', 1)
        self._add_item('dude1', 'Dude (beige)', 2)
        self._add_item('dude2', 'Dude (yellow)', 3)
        self._add_item('dude3', 'Dude (pink)', 4)
        self._add_item('dude4', 'Dude (blue)', 5)
        self._add_item('dude5', 'Dude (green)', 6)
        palette.itemSelectionChanged.connect(self._item_activated)
        palette.setCurrentRow(1)

        self._action('actionNew').triggered.connect(self._new_dialog)
        self._action('actionOpen').triggered.connect(self._open)
        self._action('actionSave').triggered.connect(self._save)
        self._action('actionSave_As').triggered.connect(self._save_as)
        self._action('actionAbout').triggered.connect(self._about)

    def _action(self, name):
        return self.win.findChild(QtWidgets.QAction, name)

    def _open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.win,
                                                        'Open maze',
                                                        self.last_dir,
                                                        NP_FILES)
        if not path:
            return
        self.last_dir = os.path.dirname(path)
        filename = os.path.basename(path)
        try:
            self.array = self.grid.array = numpy.loadtxt(path, dtype=MAZE_T)
        except BaseException as e:
            self._error_dialog('Could not open {}'.format(filename), str(e))
            return
        self.filename = filename
        self.path = path
        self._update_title()

    def _save_as(self):
        if self.path:
            path = self.path
        else:
            path = os.path.join(self.last_dir, self.filename)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self.win,
                                                        'Save maze',
                                                        path,
                                                        NP_FILES)
        if not path:
            return
        self.last_dir = os.path.dirname(path)
        filename = os.path.basename(path)
        try:
            numpy.savetxt(path, self.array)
        except BaseException as e:
            self._error_dialog('Could not save {}'.format(filename), str(e))
            return
        self.filename = filename
        self.path = path
        self._update_title()

    def _save(self):
        if not self.path:
            return self._save_as()
        try:
            numpy.savetxt(self.path, self.array)
        except BaseException as e:
            filename = os.path.basename(self.path)
            self._error_dialog('Could not save {}'.format(filename), str(e))
            return

    def _new_dialog(self):
        self.new_dialog = dialog = QtWidgets.QDialog(self.win)
        dialog.setModal(True)
        with open(get_filename('ui/newmaze.ui')) as f:
            uic.loadUi(f, dialog)
        dialog.show()
        dialog.finished.connect(self._new_finsihed)

    def _new_finsihed(self, result):
        dialog = self.new_dialog
        self.new_dialog = None
        if not result:
            dialog.destroy()
            return
        w = dialog.findChild(QtWidgets.QSpinBox, 'widthBox').value()
        h = dialog.findChild(QtWidgets.QSpinBox, 'heightBox').value()
        c = dialog.findChild(QtWidgets.QSlider, 'complexitySlider').value()
        d = dialog.findChild(QtWidgets.QSlider, 'densitySlider').value()
        dialog.destroy()

        self.array = self.grid.array = generator.maze(h, w, c/100, d/100)

        self.grid._resize()
        self.path = None
        self.filename = 'untitled.csv.gz'
        self._update_title()

    def _update_title(self):
        self.win.setWindowTitle('Maze [{}]'.format(self.filename))

    def _error_dialog(self, title, msg):
        QtWidgets.QMessageBox.critical(self.win, title, title + '\n\n' + msg)

    def _about(self):
        html = publish_parts(__doc__, writer_name='html')['html_body']
        QtWidgets.QMessageBox.about(self.win, 'About maze', html)

    def _add_item(self, name, text, number, svg=None):
        item = QtWidgets.QListWidgetItem(text)
        icon = QtGui.QIcon()
        filename = get_filename('pics/{}.svg'.format(name))
        icon.addFile(filename)
        item.setIcon(icon)
        self.palette.addItem(item)
        svg = svg or QtSvg.QSvgRenderer(filename)
        self.grid.pics[number] = svg
        item.setData(KIND_ROLE, number)

    def _item_activated(self):
        for item in self.palette.selectedItems():
            self.grid.selected_tile_kind = item.data(KIND_ROLE)

    def run(self):
        self.win.show()
        return self.app.exec_()

def main():
    gui = Gui()
    return gui.run()
