import sys
import logging
import tkinter as tk
from typing import Any

import numpy as np

from geometry import Edge, Number, Object, Point, Polygon, Vertex
from affine import TkinterCanvas


class ObjLoaderError(ValueError):
    pass


class ObjLoader:
    def __init__(self) -> None:
        self.object: Object = Object()
        self._positions: list[tuple[float, float, float, float]] = []
        self._texcoords: list[tuple[float, float, float | None]] = []
        self._normals: list[tuple[float, float, float]] = []
        self._vertex_map: dict[tuple[int, int | None, int | None], Vertex] = {}

    @staticmethod
    def _parse_obj_index(index_str: str, count: int) -> int:
        i = int(index_str)
        if i == 0:
            raise ValueError("OBJ indices are 1-based; 0 is invalid")
        return (i - 1) if i > 0 else (count + i)

    def _parse_face_vertex(self, token: str) -> tuple[int, int | None, int | None]:
        parts = token.split('/')
        if not parts or parts[0] == "":
            raise ValueError(f"Invalid face vertex token: {token!r}")

        v_idx = self._parse_obj_index(parts[0], len(self._positions))
        vt_idx: int | None = None
        vn_idx: int | None = None

        if len(parts) >= 2 and parts[1] != "":
            vt_idx = self._parse_obj_index(parts[1], len(self._texcoords))
        if len(parts) >= 3 and parts[2] != "":
            vn_idx = self._parse_obj_index(parts[2], len(self._normals))

        return v_idx, vt_idx, vn_idx

    @staticmethod
    def _compute_face_normal(face_points: list[Vertex]) -> tuple[float, float, float] | None:
        if len(face_points) < 3:
            return None

        p0 = np.array([face_points[0].x, face_points[0].y, face_points[0].z], dtype=float)
        p1 = np.array([face_points[1].x, face_points[1].y, face_points[1].z], dtype=float)
        p2 = np.array([face_points[2].x, face_points[2].y, face_points[2].z], dtype=float)

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm == 0.0:
            return None

        normal = normal / norm
        return float(normal[0]), float(normal[1]), float(normal[2])

    def load(self, filepath: str) -> None:
        if not filepath.endswith('.obj'):
            raise ObjLoaderError("Only .obj files are supported.")

        # Reset state for each load
        self.object = Object()
        self._positions = []
        self._texcoords = []
        self._normals = []
        self._vertex_map = {}

        with open(filepath, 'r') as file:
            lines = file.readlines()

        for line in lines:
            try:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'v':
                    x, y, z = map(float, parts[1:4])
                    w = float(parts[4]) if len(parts) > 4 else 1.0
                    self._positions.append((x, y, z, w))

                elif parts[0] == 'vt':
                    # OBJ: vt u v [w]
                    u = float(parts[1])
                    v = float(parts[2]) if len(parts) > 2 else 0.0
                    w = float(parts[3]) if len(parts) > 3 else None
                    self._texcoords.append((u, v, w))

                elif parts[0] == 'vn':
                    x, y, z = map(float, parts[1:4])
                    self._normals.append((x, y, z))

                elif parts[0] == 'f':
                    corner_indices = [self._parse_face_vertex(tok) for tok in parts[1:]]
                    face_points: list[Vertex] = []
                    for v_idx, vt_idx, vn_idx in corner_indices:
                        key = (v_idx, vt_idx, vn_idx)
                        vertex = self._vertex_map.get(key)
                        if vertex is None:
                            x, y, z, w = self._positions[v_idx]
                            u, v, _ = self._texcoords[vt_idx] if vt_idx is not None else (0.0, 0.0, None)
                            normal = self._normals[vn_idx] if vn_idx is not None else None
                            vertex = Vertex(x, y, z, w, u=u, v=v, normal=normal)
                            self._vertex_map[key] = vertex
                            self.object.points.append(vertex)
                        face_points.append(vertex)

                    if any(v.normal is None for v in face_points):
                        normal = self._compute_face_normal(face_points)
                        if normal is not None:
                            for v in face_points:
                                if v.normal is None:
                                    v.normal = normal

                    if len(face_points) == 2:
                        self.object.edges.append(Edge(face_points))
                    elif len(face_points) > 2:
                        edges = []
                        for i in range(len(face_points)):
                            edge = Edge([face_points[i], face_points[(i + 1) % len(face_points)]])
                            edges.append(edge)
                        self.object.polygons.append(Polygon(edges))

            except (ValueError, IndexError) as e:
                raise ObjLoaderError(f"Error parsing line: '{line.strip()}'. {e}") from e

    def __call__(self, dx: Number = 0.0, dy: Number = 0.0, dz: Number = 0.0, sx: Number = 1.0, sy: Number = 1.0, sz: Number = 1.0) -> Object:
        point_map: dict[int, Point] = {}
        new_points: list[Point] = []
        for p in self.object.points:
            if isinstance(p, Vertex):
                p2 = Vertex(
                    p.x,
                    p.y,
                    p.z,
                    getattr(p, "w", 1.0),
                    u=p.u,
                    v=p.v,
                    normal=p.normal,
                )
            else:
                p2 = Point(p.x, p.y, p.z, getattr(p, "w", 1.0))
            point_map[id(p)] = p2
            new_points.append(p2)

        new_edges: list[Edge] = [Edge([point_map[id(p)] for p in e.points]) for e in self.object.edges]

        new_polygons = [
            Polygon([Edge([point_map[id(p)] for p in e.points]) for e in poly.edges])
            for poly in self.object.polygons
        ]

        obj = Object(new_points, new_edges, new_polygons)
        if sx != 1.0 or sy != 1.0 or sz != 1.0:
            obj.scale(sx, sy, sz)
        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            obj.translate(dx, dy, dz)
        return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(object={self.object}, id={hex(id(self))})"

def main(*args: Any) -> None:
    if not args:
        args = ("./utah_teapot_lowpoly.obj",)
        # logging.error("Filepath argument is required.")
        # exit(1)

    canvas = TkinterCanvas(400, 400)

    loader = ObjLoader()
    try:
        loader.load(args[0])
    except ObjLoaderError as e:
        logging.exception(e)
        exit(1)
    object = loader(0.0, 0.0, 3)

    print(object)

    canvas += object

    canvas.draw()
    tk.mainloop()

if __name__ == "__main__":
    main(*sys.argv[1:])