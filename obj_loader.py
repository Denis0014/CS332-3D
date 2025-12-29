import sys
import logging
import tkinter as tk
from typing import Any
from geometry import ObjLoader, ObjLoaderError
from affine import TkinterCanvas

def main(*args: Any) -> None:
    if not args:
        args = ("./test.obj",)
        # logging.error("Filepath argument is required.")
        # exit(1)

    canvas = TkinterCanvas(400, 400)

    loader = ObjLoader()
    try:
        loader.load(args[0])
    except ObjLoaderError as e:
        logging.exception(e)
        exit(1)
    points, edges, polygons = loader(0.0, 0.0, 0.2)

    # print(points)

    for shape in points + edges + polygons:
        canvas += shape

    canvas.draw_shapes()
    tk.mainloop()

if __name__ == "__main__":
    main(*sys.argv[1:])