import torch
import tkinter as tk
import os

assert __name__ != '__main__', 'Module startup error.'

BACKGROUND = 'black'


class Canvas(tk.Canvas):
    def __init__(self, *args, **kwargs):
        self.cell_size = kwargs.pop('cell')
        self.matrix = torch.zeros(kwargs['height'], kwargs['width'], dtype=torch.int16)

        self.__focus = None
        self.__tool = None
        self.coordinates = None

        kwargs['height'] *= self.cell_size
        kwargs['width'] *= self.cell_size
        kwargs['background'] = BACKGROUND
        super().__init__(*args, **kwargs)

        self.bind('<Motion>', self.move_)
        self.bind('<ButtonPress-1>', lambda _e: self.detect('pen'))
        self.bind('<ButtonPress-3>', lambda _e: self.detect('eraser'))
        self.bind('<ButtonRelease>', lambda _e: self.release('any'))

    def move_(self, event):
        self.coordinates = (event.y, event.x)

    def detect(self, tool: str):
        if tool != self.__tool:
            self.release(tool)
        self.__tool = tool

        y, x = self.coordinates
        if self.__focus and 0 < y < self.winfo_height() and 0 < x < self.winfo_width():
            y //= self.cell_size
            x //= self.cell_size

            self.matrix[y, x] = max(0, min(self.matrix[y, x] + (-255, +16)[self.__tool == 'pen'], 255))
            self.draw(y, x)

            if self.__tool == 'pen':
                for index in ((0, -1), (+1, 0), (0, +1), (-1, 0)):
                    y_ = y + index[0]
                    x_ = x + index[1]

                    if 0 <= y_ * self.cell_size < self.winfo_height() and 0 <= x_ * self.cell_size < self.winfo_width():
                        self.matrix[y_, x_] = max(0, min(self.matrix[y_, x_] + 8, 255))
                        self.draw(y_, x_)

        self.__focus = self.after(2, self.detect, tool)

    def release(self, tool: str):
        if self.__focus is None:
            return

        self.after_cancel(self.__focus)
        self.__focus = None
        self.__tool = None

        if tool != 'any':
            self.detect(tool)

    def draw(self, y: int, x: int):
        self.create_rectangle(
            x * self.cell_size, y * self.cell_size,
            (x + 1) * self.cell_size, (y + 1) * self.cell_size,
            fill='#' + ('%02x' % (self.matrix[y, x].item())) * 3, width=0)

    def clear(self):
        self.delete('all')
        self.matrix = torch.zeros_like(self.matrix)


class Application:
    def __init__(self, function):
        self.root = tk.Tk()
        self.canvas = Canvas(self.root, cell=25, height=28, width=28, highlightthickness=0)
        self.canvas.pack()

        self.__customization()
        self.root.bind('<F5>', lambda _e: self.canvas.clear())
        self.root.bind('<Return>', lambda _e: function(self.canvas.matrix.detach()))

    def __customization(self):
        self.root.title('Paint')
        self.root.iconbitmap(os.path.join('interface', 'icon.ico'))
        self.root.resizable(False, False)

    def start(self):
        self.root.mainloop()
