from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Iterator, List, Literal, Tuple, Union, Any, Sequence, Self, override

Number = Union[int, float]
PointType = Union["BasePoint", "Point"]
EdgeType = Union["BaseEdge", "Edge"]
PolygonType = Union["BasePolygon", "Polygon"]
Shape = Union[PointType, EdgeType, PolygonType]


class ITransformable:
    def __init__(self) -> None:
        self.points: List[PointType] = self.get_shape_points()

    def get_shape_points(self) -> List[PointType]:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def centered(self, func: Callable[..., Self], *args: Any, **kwargs: Any) -> Self:
        center_x = sum(point.x for point in self.points) / len(self.points)
        center_y = sum(point.y for point in self.points) / len(self.points)
        center_z = sum(point.z for point in self.points) / len(self.points)

        self.translate(-center_x, -center_y, -center_z)
        result = func(*args, **kwargs)
        self.translate(center_x, center_y, center_z)
        return result

    def point_selected(self, func: Callable[..., Self], point: PointType, *args: Any, **kwargs: Any) -> Self:
        self.translate(-point.x, -point.y, -point.z) if point else None
        result = func(*args, **kwargs)
        self.translate(point.x, point.y, point.z) if point else None
        return result

    def transform(self, matrix: np.ndarray) -> Self:
        for point in self.points:
            vec = np.array([point.x, point.y, point.z, 1])
            transformed_vec = vec @ matrix
            point.x = transformed_vec[0] / transformed_vec[3]
            point.y = transformed_vec[1] / transformed_vec[3]
            point.z = transformed_vec[2] / transformed_vec[3]
        return self
    
    def projection(self, matrix: np.ndarray) -> Iterator[BasePoint]:
        for point in self.points:
            vec = np.array([point.x, point.y, point.z, 1])
            transformed_vec = matrix @ vec
            yield BasePoint(
                transformed_vec[0] / transformed_vec[3],
                transformed_vec[1] / transformed_vec[3],
                transformed_vec[2] / transformed_vec[3],
            )

    def scale(self, sx: Number, sy: Number, sz: Number) -> Self:
        scale_matrix = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])
        return self.transform(scale_matrix)
    
    def scale_from_center(self, sx: Number, sy: Number, sz: Number) -> Self:
        return self.centered(self.scale, sx, sy, sz)
    
    def scale_from_point(self, point: PointType, sx: Number, sy: Number, sz: Number) -> Self:
        return self.point_selected(self.scale, point, sx, sy, sz)
    
    def translate(self, tx: Number, ty: Number, tz: Number) -> Self:
        translation_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [tx, ty, tz, 1]
        ])
        return self.transform(translation_matrix)
    
    def rotate(self, angle_x: Number, angle_y: Number, angle_z: Number) -> Self:
        rx = np.radians(angle_x)
        ry = np.radians(angle_y)
        rz = np.radians(angle_z)

        rotation_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), np.sin(rx), 0],
            [0, -np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ])

        rotation_y = np.array([
            [np.cos(ry), 0, -np.sin(ry), 0],
            [0, 1, 0, 0],
            [np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ])

        rotation_z = np.array([
            [np.cos(rz), np.sin(rz), 0, 0],
            [-np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        rotation_matrix = rotation_z @ rotation_y @ rotation_x
        return self.transform(rotation_matrix)
    
    def reflect(self, axis: Literal['x', 'y', 'z']) -> Self:
        if axis == 'x':
            reflect_matrix = np.array([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'y':
            reflect_matrix = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'z':
            reflect_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        return self.transform(reflect_matrix)


class BaseShape(type):
    pass


class BasePoint(metaclass=BaseShape):
    def __init__(self, x: Number, y: Number, z: Number) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.x}, {self.y}, {self.z}, id={hex(id(self))})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BasePoint):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))
    
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z


class Point(BasePoint, ITransformable):
    def __init__(self, x: Number, y: Number, z: Number, w: Number = 1.0) -> None:
        BasePoint.__init__(self, x, y, z)
        ITransformable.__init__(self)
        self.w = w
        self.__features: Dict[str, Any] = dict()

    @override
    def get_shape_points(self) -> List[PointType]:
        return [self]

    def __getitem__(self, item: str) -> Any:
        return self.__features.get(item, None)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.__features[key] = value

    def __repr__(self) -> str:
        features = ", ".join(f"{k}={repr(v)}" for k, v in self.__features.items())
        if features:
            return f"{self.__class__.__name__}({self.x}, {self.y}, {self.z}, {features}, id={hex(id(self))})"
        
        else:
            return f"{self.__class__.__name__}({self.x}, {self.y}, {self.z}, id={hex(id(self))})"


class BaseEdge(metaclass=BaseShape):
    def __init__(self, points: Sequence[PointType]) -> None:
        self.points: List[PointType] = list(points)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.points}, id={hex(id(self))})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseEdge):
            return False
        return self.points == other.points
    
    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)
    
    def __add__(self, other: Union[PointType, EdgeType]) -> Self:
        if isinstance(other, BaseEdge):
            return self.__class__(**self.__dict__, points=self.points + other.points)
        
        else:
            return self.__class__(**self.__dict__, points=self.points + [other])
        
    def __iadd__(self, other: Union[PointType, EdgeType]) -> Self:
        if isinstance(other, BaseEdge):
            self.points.extend(other.points)

        else:
            self.points.append(other)

        return self
    
    def __hash__(self) -> int:
        return hash(tuple(self.points))
    
    def __len__(self) -> int:
        return len(self.points)
    
    def __iter__(self):
        for point in self.points:
            yield point


class Edge(BaseEdge, ITransformable):
    def __init__(self, points: Sequence[PointType]) -> None:
        BaseEdge.__init__(self, points)
        ITransformable.__init__(self)
        self.__features: Dict[str, Any] = dict()

    @override
    def get_shape_points(self) -> List[PointType]:
        return self.points
    
    def __getitem__(self, item: str) -> Any:
        return self.__features.get(item, None)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.__features[key] = value

    def __repr__(self) -> str:
        features = ", ".join(f"{k}={repr(v)}" for k, v in self.__features.items())
        if features:
            return f"{self.__class__.__name__}({self.points}, {features}, id={hex(id(self))})"
        
        else:
            return f"{self.__class__.__name__}({self.points})"


class BasePolygon(metaclass=BaseShape):
    def __init__(self, edges: Sequence[EdgeType]) -> None:
        self.edges: List[EdgeType] = list(edges)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.edges}, id={hex(id(self))})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BasePolygon):
            return False
        return self.edges == other.edges
    
    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)
    
    def __len__(self) -> int:
        return len(self.edges)
    
    def __add__(self, other: EdgeType) -> Self:
        return self.__class__(**self.__dict__, edges=self.edges + [other])
    
    def __iadd__(self, other: EdgeType) -> Self:
        self.edges.append(other)
        return self
    
    def __hash__(self) -> int:
        return hash(tuple(self.edges))
    
    def __iter__(self):
        for edge in self.edges:
            yield edge


class Polygon(BasePolygon, ITransformable):
    def __init__(self, edges: Sequence[EdgeType]) -> None:
        BasePolygon.__init__(self, edges)
        ITransformable.__init__(self)
        self.__features: Dict[str, Any] = dict()

    @override
    def get_shape_points(self) -> List[PointType]:
        points: List[PointType] = []
        for edge in self.edges:
            points.extend(edge.points)
        return points
    
    def __getitem__(self, item: str) -> Any:
        return self.__features.get(item, None)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.__features[key] = value

    def __repr__(self) -> str:
        features = ", ".join(f"{k}={repr(v)}" for k, v in self.__features.items())
        if features:
            return f"{self.__class__.__name__}({self.edges}, {features}, id={hex(id(self))})"
        
        else:
            return f"{self.__class__.__name__}({self.edges}, id={hex(id(self))})"


class BaseCanvas(ITransformable):
    def __init__(
            self,
            width: int,
            height: int,
            points: Sequence[PointType] = (),
            edges: Sequence[EdgeType] = (),
            polygons: Sequence[PolygonType] = (),
        ) -> None:
        self.width = width
        self.height = height
        self.points: List[PointType] = list(points)
        self.edges: List[EdgeType] = list(edges)
        self.polygons: List[PolygonType] = list(polygons)

    @override
    def get_shape_points(self) -> List[PointType]:
        return self.points

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(width={self.width}, height={self.height}, "
            f"polygons={self.polygons}, id={hex(id(self))})"
        )
    
    def __add__(self, shape: Shape) -> Self:
        new = self.__class__(**self.__dict__)
        if (isinstance(point := shape, BasePoint)):
            if point not in new.points:
                new.points.append(point)
        elif isinstance(edge := shape, BaseEdge):
            for point in edge:
                new = new.__add__(point)
            if edge not in new.edges:
                new.edges.append(edge)
        elif isinstance(polygon := shape, BasePolygon):
            for edge in polygon:
                new = new.__add__(edge)
            if polygon not in new.polygons:
                new.polygons.append(polygon)
        return new
    
    def __iadd__(self, shape: Shape) -> Self:
        if (isinstance(point := shape, BasePoint)):
            if point not in self.points:
                self.points.append(point)
        elif isinstance(edge := shape, BaseEdge):
            for point in edge:
                self.__iadd__(point)
            if edge not in self.edges:
                self.edges.append(edge)
        elif isinstance(polygon := shape, BasePolygon):
            for edge in polygon:
                self.__iadd__(edge)
            if polygon not in self.polygons:
                self.polygons.append(polygon)
        return self
    
    def __sub__(self, shape: Shape) -> Self:
        new = self.__class__(**self.__dict__)
        if isinstance(shape, BasePoint):
            new.points.remove(shape)
        elif isinstance(shape, BaseEdge):
            new.edges.remove(shape)
        else:
            new.polygons.remove(shape)
        return new
    
    def __isub__(self, shape: Shape) -> Self:
        if isinstance(shape, BasePoint):
            self.points.remove(shape)
        elif isinstance(shape, BaseEdge):
            self.edges.remove(shape)
        else:
            self.polygons.remove(shape)
        return self
    
    def clear(self) -> None:
        self.points.clear()
        self.edges.clear()
        self.polygons.clear()


class ObjLoaderError(ValueError):
    pass

class ObjLoader:
    def __init__(self) -> None:
        self.points: List[Point] = []
        self.edges: List[Edge] = []
        self.polygons: List[Polygon] = []
    
    def load(self, filepath: str) -> None:
        if not filepath.endswith('.obj'):
            raise ObjLoaderError("Only .obj files are supported.")

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
                    self.points.append(Point(x, y, z, w))

                elif parts[0] == 'vt':
                    continue

                elif parts[0] == 'vn':
                    continue
                
                elif parts[0] == 'f':
                    vertex_indices = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                    face_points = [self.points[idx] for idx in vertex_indices]
                    
                    if len(face_points) == 2:
                        self.edges.append(Edge(face_points))
                    elif len(face_points) > 2:
                        edges = []
                        for i in range(len(face_points)):
                            edge = Edge([face_points[i], face_points[(i + 1) % len(face_points)]])
                            edges.append(edge)
                        self.polygons.append(Polygon(edges))

            except (ValueError, IndexError) as e:
                raise ObjLoaderError(f"Error parsing line: '{line.strip()}'. {e}") from e
    
    def __call__(self, dx: Number = 0.0, dy: Number = 0.0, dz: Number = 0.0) -> Tuple[List[Point], List[Edge], List[Polygon]]:
        points, edges, polygons = self.points, self.edges, self.polygons

        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            for point in points:
                point.x += dx
                point.y += dy
                point.z += dz

            for edge in edges:
                for point in edge.points:
                    point.x += dx
                    point.y += dy
                    point.z += dz
            
            for polygon in polygons:
                for edge in polygon.edges:
                    for point in edge.points:
                        point.x += dx
                        point.y += dy
                        point.z += dz

        return points, edges, polygons
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(points={len(self.points)}, edges={len(self.edges)}, polygons={len(self.polygons)}, id={hex(id(self))})"