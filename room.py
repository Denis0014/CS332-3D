import tkinter as tk
from typing import List
import numpy as np

from geometry import Object, Vertex, Edge, Polygon
from obj_loader import ObjLoader
from shader import Material, PointLight
from affine import TkinterCanvas

loader = ObjLoader()


def _face_normal(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
	v1 = p1 - p0
	v2 = p2 - p0
	n = np.cross(v1, v2)
	norm = np.linalg.norm(n)
	if norm == 0.0:
		return (0.0, 0.0, 0.0)
	n = n / norm
	return (float(n[0]), float(n[1]), float(n[2]))


def _add_quad(
	obj: Object,
	p0: tuple[float, float, float],
	p1: tuple[float, float, float],
	p2: tuple[float, float, float],
	p3: tuple[float, float, float],
	material: Material,
	normal: tuple[float, float, float] | None = None,
) -> None:
	if normal is None:
		n = _face_normal(np.array(p0), np.array(p1), np.array(p2))
	else:
		n = normal

	v0 = Vertex(*p0, normal=n)
	v1 = Vertex(*p1, normal=n)
	v2 = Vertex(*p2, normal=n)
	v3 = Vertex(*p3, normal=n)

	e0 = Edge([v0, v1])
	e1 = Edge([v1, v2])
	e2 = Edge([v2, v3])
	e3 = Edge([v3, v0])
	poly = Polygon([e0, e1, e2, e3])
	poly["material"] = material

	obj.points.extend([v0, v1, v2, v3])
	obj.edges.extend([e0, e1, e2, e3])
	obj.polygons.append(poly)


def _add_box(
	obj: Object,
	dx: float,
	dy: float,
	dz: float,
	sx: float,
	sy: float,
	sz: float,
	material: Material,
) -> None:
	loader.load("cube.obj")
	cube = loader(
		dx, dy, dz, sx, sy, sz
	)
	if cube is None:
		return

	for poly in cube.polygons:
		if isinstance(poly, Polygon):
			poly["material"] = material

	obj.points.extend(cube.points)
	obj.edges.extend(cube.edges)
	obj.polygons.extend(cube.polygons)


def build_cornell_room() -> List[Object]:
	room = Object()
	box1 = Object()
	box2 = Object()

	white = Material(
		ambient=(0.9, 0.9, 0.9),
		diffuse=(0.9, 0.9, 0.9),
		specular=(0.1, 0.1, 0.1),
		shininess=8.0,
	)
	red = Material(
		ambient=(0.8, 0.1, 0.1),
		diffuse=(0.8, 0.1, 0.1),
		specular=(0.05, 0.05, 0.05),
		shininess=4.0,
	)
	green = Material(
		ambient=(0.1, 0.8, 0.1),
		diffuse=(0.1, 0.8, 0.1),
		specular=(0.05, 0.05, 0.05),
		shininess=4.0,
	)

	# Room dimensions (push everything in front of the camera)
	x0, x1 = -1.0, 1.0
	y0, y1 = -1.0, 1.0
	z0, z1 = 1.0, 6.0

	# Floor (+y normal for inside)
	_add_quad(
		room,
		(x0, y0, z0),
		(x1, y0, z0),
		(x1, y0, z1),
		(x0, y0, z1),
		white,
		normal=(0.0, 1.0, 0.0),
	)
	# Ceiling (-y normal for inside)
	_add_quad(
		room,
		(x0, y1, z1),
		(x1, y1, z1),
		(x1, y1, z0),
		(x0, y1, z0),
		white,
		normal=(0.0, -1.0, 0.0),
	)
	# Back wall (-z normal for inside)
	_add_quad(
		room,
		(x0, y0, z1),
		(x1, y0, z1),
		(x1, y1, z1),
		(x0, y1, z1),
		white,
		normal=(0.0, 0.0, -1.0),
	)
	# Left wall (+x normal for inside)
	_add_quad(
		room,
		(x0, y0, z1),
		(x0, y0, z0),
		(x0, y1, z0),
		(x0, y1, z1),
		red,
		normal=(1.0, 0.0, 0.0),
	)
	# Right wall (-x normal for inside)
	_add_quad(
		room,
		(x1, y0, z0),
		(x1, y0, z1),
		(x1, y1, z1),
		(x1, y1, z0),
		green,
		normal=(-1.0, 0.0, 0.0),
	)

	# Two boxes
	_add_box(box1, 0.4, -0.9, 3, 0.3, 0.3, 0.3, white)
	_add_box(box2, -0.6, 0.9, 3.2, 0.3, 0.6, 0.2, red)

	return [room, box1, box2]

def main() -> None:
	light = PointLight(
		position=(0.0, 1.9, 3.5),
		ambient=(0.15, 0.15, 0.15),
		diffuse=(1.0, 1.0, 1.0),
		specular=(1.0, 1.0, 1.0),
		attenuation=(1.0, 0.08, 0.02),
	)
	canvas = TkinterCanvas(700, 700, light=light, view_pos=(0.0, 1.5, -4.0))
	room = build_cornell_room() 
	for obj in room:
		canvas += obj
	canvas.draw()
	tk.mainloop()


if __name__ == "__main__":
	main()

