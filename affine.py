import tkinter
import tkinter.messagebox
import numpy as np
from tkinter import simpledialog
from geometry import BaseCanvas, Edge, ITransformable, Object, ObjectType, Point, PointType, Polygon
from shader import FragmentShader, PointLight, VertexShader
from typing import Any, Callable, Iterator, Optional, Sequence

FOV = 60  # Field of view in degrees
FAR = 1000.0
NEAR = 0.1

class TkinterCanvas(BaseCanvas):
    def __init__(self, width: int, height: int, light: PointLight | None = None, view_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        super().__init__(width, height)
        self.tk_canvas = tkinter.Canvas(width=width, height=height)
        self.tk_canvas.pack()
        root = self.tk_canvas.winfo_toplevel()
        self.create_menubar(root)
        self.last_action: dict[str, Any] = {}
        self.light = light
        self.view_pos = view_pos

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

        a = 1.0 / (aspect * np.tan(fov / 2))
        b = 1.0 / np.tan(fov / 2)
        c = FAR / (FAR - NEAR)
        d = -FAR * NEAR / (FAR - NEAR)
        matrix = np.array([
            [a, 0, 0, 0],
            [0, b, 0, 0],
            [0, 0, c, 1],
            [0, 0, d, 0],
        ], dtype=float)

        for point in points:
            if point.z <= 0:
                continue

            vec = np.array([point.x, point.y, point.z, 1])
            transformed_vec = vec @ matrix
            if transformed_vec[3] != 0:
                x_ndc = transformed_vec[0] / transformed_vec[3]
                y_ndc = transformed_vec[1] / transformed_vec[3]
            else:
                x_ndc = transformed_vec[0]
                y_ndc = transformed_vec[1]

            x_screen = (x_ndc + 1) * 0.5 * width
            y_screen = (1 - y_ndc) * 0.5 * height

            yield Point(x_screen, y_screen, 0)
    
    def draw(self) -> None:
        self.tk_canvas.delete("all")

        vertex_shader = VertexShader()
        fragment_shader = FragmentShader(light=self.light, view_pos=self.view_pos)

        for polygon in sorted(
            self.polygons,
            key=lambda poly: sum(point.z for edge in poly for point in edge.points) / max(1, sum(len(edge.points) for edge in poly)),
            reverse=True,
        ):
            ordered_points = []
            seen: set[int] = set()
            for edge in polygon:
                if not edge.points:
                    continue
                point = edge.points[0]
                pid = id(point)
                if pid in seen:
                    continue
                seen.add(pid)
                ordered_points.append(point)
                
            if len(ordered_points) < 3:
                continue

            shader_outputs = [vertex_shader(point) for point in ordered_points]
            world_points = []
            normals = []
            for out in shader_outputs:
                w = out.position[3] if out.position.shape[0] > 3 else 1.0
                world_points.append(Point(out.position[0] / w, out.position[1] / w, out.position[2] / w))
                if out.normal is not None:
                    normals.append(out.normal)

            if len(world_points) < 3:
                continue
            
            # Geometric normal for culling
            p0 = np.array([world_points[0].x, world_points[0].y, world_points[0].z], dtype=float)
            p1 = np.array([world_points[1].x, world_points[1].y, world_points[1].z], dtype=float)
            p2 = np.array([world_points[2].x, world_points[2].y, world_points[2].z], dtype=float)
            v1 = p1 - p0
            v2 = p2 - p0
            cross = np.cross(v1, v2)
            norm = np.linalg.norm(cross)
            face_normal = cross / norm if norm != 0.0 else None

            # # Back-face culling
            # if face_normal is not None:
            #     view_vec = p0 - np.array(self.view_pos)
            #     if np.dot(face_normal, view_vec) >= 0:
            #         continue

            projected = list(self.projection_onto_2d(world_points, self.width, self.height))
            if len(projected) < 3:
                continue

            normal_vec = None
            if normals:
                normal_vec = np.mean(normals, axis=0)
            elif face_normal is not None:
                normal_vec = face_normal

            avg_pos = None
            if world_points:
                avg_pos = np.mean(
                    [[p.x, p.y, p.z] for p in world_points],
                    axis=0,
                )

            material = None
            
            if isinstance(polygon, Polygon):
                material = polygon["material"]

            fill_color = FragmentShader.rgb_to_hex(
                fragment_shader.shade(avg_pos, normal_vec, material=material)
            )

            coords = [coord for point in projected for coord in (point.x, point.y)]
            self.tk_canvas.create_polygon(coords, outline='black', fill=fill_color, width=1)

        # for point in self.projection_onto_2d(self.points, self.width, self.height):
        #     self.tk_canvas.create_oval(
        #         point.x - 2, point.y - 2,
        #         point.x + 2, point.y + 2,
        #         fill='black'
        #     )

    def clear(self) -> None:
        super().clear()
        self.tk_canvas.delete("all")

    def get_object(self) -> Optional[ObjectType]:
        num = simpledialog.askinteger(
            "Input for object", 
            "Enter object number:"
        )

        if num is None or not self.objects or len(self.objects) < num or num < 1:
            tkinter.messagebox.showerror(
                "Invalid object",
                "Invalid object number."
            )
            return None
        
        return self.objects[num - 1]
    
    def highlight_object(self, obj: ObjectType) -> None:
        for edges in obj.polygons:
            for edge in edges:
                for p1, p2 in zip(
                    self.projection_onto_2d(edge.points, self.width, self.height),
                    self.projection_onto_2d(edge.points[1:], self.width, self.height)
                ):
                    self.tk_canvas.create_line(p1.x, p1.y, p2.x, p2.y, fill='red', width=1, dash=(4, 2))

    @staticmethod
    def with_point_selection(func: Callable[..., Any]) -> Callable[..., Any]:
        def ask() -> Iterator[Optional[float]]:
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
        def ask() -> Iterator[Optional[float]]:
            for axis in ['sx', 'sy', 'sz']:
                yield simpledialog.askfloat(
                    f"Input for {axis}", 
                    f"Enter scaling factor for {axis}-axis:"
                )

        try:
            obj = kwargs.get("object") or self.get_object()
            if obj is None or not isinstance(obj, Object):
                tkinter.messagebox.showerror(
                    "Invalid object",
                    "Selected object is not transformable."
                )
                return
            self.highlight_object(obj)
            
            sx, sy, sz = kwargs.get("factors") or ask()
            if sx is None or sy is None or sz is None:
                tkinter.messagebox.showerror(
                    "Invalid scaling factors",
                    "Invalid scaling factors. Please enter valid numbers for all axes."
                )
                return

            obj.scale_from_center(sx, sy, sz)
            self.last_action = {
                "action": self.apply_scale,
                "kwargs": {"object": obj, "factors": (sx, sy, sz)}
            }
        finally:
            self.draw()

    def apply_translation(self, **kwargs: Any) -> None:
        def ask() -> Iterator[Optional[float]]:
            for axis in ['tx', 'ty', 'tz']:
                yield simpledialog.askfloat(
                    f"Input for {axis}", 
                    f"Enter translation distance for {axis}-axis:"
                )
        
        try:
            obj = kwargs.get("object") or self.get_object()
            if obj is None or not isinstance(obj, Object):
                tkinter.messagebox.showerror(
                    "Invalid object",
                    "Selected object is not transformable."
                )
                return
            self.highlight_object(obj)
            
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
            self.draw()

    def apply_rotation_around_center(self, **kwargs: Any) -> None:
        def ask() -> Iterator[Optional[float]]:
            for axis in ['angle_x', 'angle_y', 'angle_z']:
                yield simpledialog.askfloat(
                    f"Input for {axis}", 
                    f"Enter rotation angle (in degrees) around {axis}-axis:"
                )

        try:
            obj = kwargs.get("object") or self.get_object()
            if obj is None or not isinstance(obj, Object):
                tkinter.messagebox.showerror(
                    "Invalid object",
                    "Selected object is not transformable."
                )
                return
            self.highlight_object(obj)
                    
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
            self.draw()

    @with_point_selection
    def apply_rotation_around_point(self, **kwargs: Any) -> None:
        def ask() -> Iterator[Optional[float]]:
            for axis in ['angle_x', 'angle_y', 'angle_z']:
                yield simpledialog.askfloat(
                    f"Input for {axis}", 
                    f"Enter rotation angle (in degrees) around {axis}-axis:"
                )

        try:
            obj = kwargs.get("object") or self.get_object()
            if obj is None or not isinstance(obj, Object):
                tkinter.messagebox.showerror(
                    "Invalid object",
                    "Selected object is not transformable."
                )
                return
            self.highlight_object(obj)
                    
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
            self.draw()

    def apply_reflection(self, **kwargs: Any) -> None:
        try:
            obj = kwargs.get("object") or self.get_object()
            if obj is None or not isinstance(obj, Object):
                tkinter.messagebox.showerror(
                    "Invalid object",
                    "Selected object is not transformable."
                )
                return
            self.highlight_object(obj)

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
            self.draw()

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

    canvas += Object(
        polygons=[
            Polygon([
                Edge([Point(-1, -1, 5), Point(1, -1, 5)]),
                Edge([Point(1, -1, 5), Point(1, 1, 5)]),
                Edge([Point(1, 1, 5), Point(-1, 1, 5)]),
                Edge([Point(-1, 1, 5), Point(-1, -1, 5)]),
            ]),
            Polygon([
                Edge([Point(-1, -1, 10), Point(1, -1, 10)]),
                Edge([Point(1, -1, 10), Point(1, 1, 10)]),
                Edge([Point(1, 1, 10), Point(-1, 1, 10)]),
                Edge([Point(-1, 1, 10), Point(-1, -1, 10)]),
            ]),
            Polygon([
                Edge([Point(-1, -1, 5), Point(-1, -1, 10)]),
                Edge([Point(1, -1, 5), Point(1, -1, 10)]),
                Edge([Point(1, 1, 5), Point(1, 1, 10)]),
                Edge([Point(-1, 1, 5), Point(-1, 1, 10)]),
            ])
        ]
    )

    canvas.draw()
    tkinter.mainloop()