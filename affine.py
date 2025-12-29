import tkinter, tkinter.messagebox
import numpy as np
from tkinter import simpledialog
from geometry import BaseCanvas, Edge, ITransformable, Point, PointType, Polygon, PolygonType
from typing import Any, Callable, Iterator, Sequence

FOV = 120  # Field of view in degrees

class TkinterCanvas(BaseCanvas):
    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self.tk_canvas = tkinter.Canvas(width=width, height=height)
        self.tk_canvas.pack()
        root = self.tk_canvas.winfo_toplevel()
        self.create_menubar(root)
        self.last_action: dict[str, Any] = {}

    def create_menubar(self, root: tkinter.Tk | tkinter.Toplevel) -> None:
        menubar = tkinter.Menu(root)
        file_menu = tkinter.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        transformations_menu = tkinter.Menu(menubar, tearoff=1)
        transformations_menu.add_command(label="Scale", command=self.apply_scale)
        transformations_menu.add_command(label="Translate", command=self.apply_translation)
        transformations_menu.add_command(label="Rotate Around Point", command=self.apply_rotation_around_point)
        transformations_menu.add_command(label="Rotate Around Center", command=self.apply_rotation_around_center)
        transformations_menu.add_command(label="Reflect", command=self.apply_reflection)
        transformations_menu.add_separator()
        transformations_menu.add_command(label="Repeat Last Action", command=self.repeat_last_action)
        menubar.add_cascade(label="Transformations", menu=transformations_menu)
        root.config(menu=menubar)
    
    @staticmethod
    def projection_onto_2d(points: Sequence[PointType], width: int, height: int) -> Iterator[Point]:
        aspect = width / height
        fov = np.radians(FOV)

        matrix = np.array([
            [1 / (aspect * np.tan(fov / 2)), 0, 0, 0],
            [0, 1 / np.tan(fov / 2), 0, 0],
            [0, 0, 1, -1],
            [0, 0, 1, 0]
        ], dtype=float)

        for point in points:
            vec = np.array([point.x, point.y, point.z, 1])
            transformed_vec = vec @ matrix
            if transformed_vec[3] != 0:
                x_ndc = transformed_vec[0] / transformed_vec[3]
                y_ndc = transformed_vec[1] / transformed_vec[3]
            else:
                x_ndc = transformed_vec[0]
                y_ndc = transformed_vec[1]

            x_screen = (1 - x_ndc) * 0.5 * width
            y_screen = (y_ndc + 1) * 0.5 * height

            yield Point(x_screen, y_screen, 0)
    
    def draw_shapes(self) -> None:
        self.tk_canvas.delete("all")

        for point in self.projection_onto_2d(self.points, self.width, self.height):
            self.tk_canvas.create_oval(
                point.x - 2, point.y - 2,
                point.x + 2, point.y + 2,
                fill='black'
            )

        for edge in self.edges:
            for p1, p2 in zip(
                self.projection_onto_2d(edge.points, self.width, self.height),
                self.projection_onto_2d(edge.points[1:], self.width, self.height)
            ):
                self.tk_canvas.create_line(p1.x, p1.y, p2.x, p2.y, fill='black')

        # for polygon in self.polygons:
        #     points = [(point.x, point.y) for edge in polygon for point in self.projection_onto_2d(edge.points)]
        #     self.tk_canvas.create_polygon(points, outline='black', fill='', width=1)

    def clear(self) -> None:
        super().clear()
        self.tk_canvas.delete("all")

    def get_polygon(self) -> PolygonType | None:
        num = simpledialog.askinteger(
            "Input for polygon", "Enter number of polygon"
        )

        if num is None or not self.polygons or len(self.polygons) < num or num < 1:
            tkinter.messagebox.showerror(
                "Invalid polygon",
                "Invalid polygon number."
            )
            return None
        
        return self.polygons[num - 1]
    
    def highlight_polygon(self, polygon: PolygonType) -> None:
        for edge in polygon:
            for p1, p2 in zip(
                self.projection_onto_2d(edge.points, self.width, self.height),
                self.projection_onto_2d(edge.points[1:], self.width, self.height)
            ):
                self.tk_canvas.create_line(p1.x, p1.y, p2.x, p2.y, fill='red', width=1, dash=(4, 2))

    @staticmethod
    def with_point_selection(func: Callable[..., Any]) -> Callable[..., Any]:
        def ask() -> Iterator[float | None]:
            for axis in ['x', 'y', 'z']:
                yield simpledialog.askfloat(
                    f"Input for {axis}-coordinate", 
                    f"Enter {axis}-coordinate of the point:"
                )
                
        def wrapper(obj: ITransformable, *args: Any, **kwargs: Any) -> Any:
            x, y, z = ask()
            if x is None or y is None or z is None:
                return
            
            point = Point(x, y, z)
            return obj.point_selected(func, point, *[obj, *args], **kwargs)
        
        return wrapper
    
    def apply_scale(self, **kwargs: Any) -> None:
        def ask() -> Iterator[float | None]:
            for axis in ['sx', 'sy', 'sz']:
                yield simpledialog.askfloat(
                    f"Input for {axis}", 
                    f"Enter scaling factor for {axis}-axis:"
                )

        try:
            obj = kwargs.get("object") or self.get_polygon()
            if obj is None or not isinstance(obj, Polygon):
                tkinter.messagebox.showerror(
                    "Invalid polygon",
                    "Selected polygon is not transformable."
                )
                return
            self.highlight_polygon(obj)
            
            sx, sy, sz = kwargs.get("factors") or ask()
            if sx is None or sy is None or sz is None:
                tkinter.messagebox.showerror(
                    "Invalid scaling factors",
                    "Invalid scaling factors. Please enter valid numbers for all axes."
                )
                return

            obj.scale_from_center(sx, sy, sz)
            self.last_action = {
                "action": self.scale_from_center,
                "kwargs": {"object": obj, "factors": (sx, sy, sz)}
            }
        finally:
            self.draw_shapes()

    def apply_translation(self, **kwargs: Any) -> None:
        def ask() -> Iterator[float | None]:
            for axis in ['tx', 'ty', 'tz']:
                yield simpledialog.askfloat(
                    f"Input for {axis}", 
                    f"Enter translation distance for {axis}-axis:"
                )
        
        try:
            obj = kwargs.get("object") or self.get_polygon()
            if obj is None or not isinstance(obj, Polygon):
                tkinter.messagebox.showerror(
                    "Invalid polygon",
                    "Selected polygon is not transformable."
                )
                return
            self.highlight_polygon(obj)
            
            tx, ty, tz = ask()
            if tx is None or ty is None or tz is None:
                tkinter.messagebox.showerror(
                    "Invalid translation",
                    "Invalid translation distances. Please enter valid numbers for all axes."
                )
                return
            
            obj.translate(tx, ty, tz)
            self.last_action = {
                "action": self.apply_translation,
                "kwargs": {"object": obj, "distances": (tx, ty, tz)}
            }
        finally:
            self.draw_shapes()

    def apply_rotation_around_center(self, **kwargs: Any) -> None:
        def ask() -> Iterator[float | None]:
            for axis in ['angle_x', 'angle_y', 'angle_z']:
                yield simpledialog.askfloat(
                    f"Input for {axis}", 
                    f"Enter rotation angle (in degrees) around {axis}-axis:"
                )

        try:
            obj = kwargs.get("object") or self.get_polygon()
            if obj is None or not isinstance(obj, Polygon):
                tkinter.messagebox.showerror(
                    "Invalid polygon",
                    "Selected polygon is not transformable."
                )
                return
            self.highlight_polygon(obj)
                    
            angle_x, angle_y, angle_z = kwargs.get("angles") or ask()
            if angle_x is None or angle_y is None or angle_z is None:
                tkinter.messagebox.showerror(
                    "Invalid angles",
                    "Invalid rotation angles. Please enter valid numbers for all axes."
                )
                return
            
            obj.rotate(angle_x, angle_y, angle_z)
            self.last_action = {
                "action": self.apply_rotation_around_center,
                "kwargs": {"object": obj, "angles": (angle_x, angle_y, angle_z)}
            }
        finally:
            self.draw_shapes()

    @with_point_selection
    def apply_rotation_around_point(self, **kwargs: Any) -> None:
        def ask() -> Iterator[float | None]:
            for axis in ['angle_x', 'angle_y', 'angle_z']:
                yield simpledialog.askfloat(
                    f"Input for {axis}", 
                    f"Enter rotation angle (in degrees) around {axis}-axis:"
                )

        try:
            obj = kwargs.get("object") or self.get_polygon()
            if obj is None or not isinstance(obj, Polygon):
                tkinter.messagebox.showerror(
                    "Invalid polygon",
                    "Selected polygon is not transformable."
                )
                return
            self.highlight_polygon(obj)
                    
            angle_x, angle_y, angle_z = kwargs.get("angles") or ask()
            if angle_x is None or angle_y is None or angle_z is None:
                tkinter.messagebox.showerror(
                    "Invalid angles",
                    "Invalid rotation angles. Please enter valid numbers for all axes."
                )
                return
            
            obj.rotate(angle_x, angle_y, angle_z)
            self.last_action = {
                "action": self.apply_rotation_around_point,
                "kwargs": {"object": obj, "angles": (angle_x, angle_y, angle_z)}
            }
        finally:
            self.draw_shapes()

    def apply_reflection(self, **kwargs: Any) -> None:
        try:
            obj = kwargs.get("object") or self.get_polygon()
            if obj is None or not isinstance(obj, Polygon):
                tkinter.messagebox.showerror(
                    "Invalid polygon",
                    "Selected polygon is not transformable."
                )
                return
            self.highlight_polygon(obj)

            axis = kwargs.get("axis") or simpledialog.askstring(
                "Input for reflection", 
                "Enter axis of reflection (x, y, or z):"
            )
            if axis not in ('x', 'y', 'z'):
                tkinter.messagebox.showerror(
                    "Invalid axis",
                    "Invalid axis of reflection. Please enter 'x', 'y' or 'z'."
                )
                return
            
            obj.reflect(axis)
            self.last_action = {
                "action": self.apply_reflection,
                "kwargs": {"object": obj, "axis": axis}
            }
        finally:
            self.draw_shapes()

    def repeat_last_action(self) -> None:
        if not self.last_action:
            tkinter.messagebox.showinfo(
                "No action to repeat",
                "There is no last action to repeat."
            )
            return
        
        action = self.last_action.get("action")
        kwargs = self.last_action.get("kwargs", {})
        if action:
            action(**kwargs)

if __name__ == "__main__":
    canvas = TkinterCanvas(400, 400)

    canvas += Polygon([
        Edge([
            Point(-30, -30, 50),
            Point(30, -30, 50),
            Point(30, 30, 50),
            Point(-30, 30, 50),
            Point(-30, -30, 50)
        ]),
        Edge([
            Point(-30, -30, 250),
            Point(30, -30, 250),
            Point(30, 30, 250),
            Point(-30, 30, 250),
            Point(-30, -30, 250)
        ]),
        Edge([
            Point(-30, -30, 50),
            Point(-30, -30, 250)
        ]),
        Edge([
            Point(30, -30, 50),
            Point(30, -30, 250)
        ]),
        Edge([
            Point(30, 30, 50),
            Point(30, 30, 250)
        ]),
        Edge([
            Point(-30, 30, 50),
            Point(-30, 30, 250)
        ])
    ])

    # canvas += Polygon([
    # Edge([
    #     Point(1, 0, 1),
    #     Point(2, 40, 50),
    #     Point(50, 50, 50),
    #     Point(40, 40, 50)
    # ]),
    # Edge([
    #     Point(40, 40, 50),
    #     Point(42, 40, 50),
    # ]),
    # Edge([
    #     Point(50, 40, 50),
    #     Point(42, 40, 50),
    # ]),
    # Edge([
    #     Point(50, 50, 50),
    #     Point(42, 40, 50),
    # ]),
    # ])

    canvas.draw_shapes()
    tkinter.mainloop()