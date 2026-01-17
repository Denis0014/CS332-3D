from __future__ import annotations
import copy
import numpy as np
from typing import Callable, Dict, Iterator, List, Literal, Union, Any, Sequence, Self

try:
    from typing import override  # type: ignore[attr-defined]
except ImportError:  # Python < 3.12
    def override(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

Number = Union[int, float]
PointType = Union["BasePoint", "Point", "Vertex"]
EdgeType = Union["BaseEdge", "Edge"]
PolygonType = Union["BasePolygon", "Polygon"]
ObjectType = Union["BaseObject", "Object"]
Shape = Union[PointType, EdgeType, PolygonType]


class ITransformable:
    _W_EPS: float = 1e-12

    def __init__(self) -> None:
        pass

    def get_shape_points(self) -> List[PointType]:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _unique_shape_points(self) -> List[PointType]:
        seen: set[int] = set()
        unique: List[PointType] = []
        for point in self.get_shape_points():
            pid = id(point)
            if pid in seen:
                continue
            seen.add(pid)
            unique.append(point)
        return unique
    
    def centered(self, func: Callable[..., Self], *args: Any, **kwargs: Any) -> Self:
        points = self._unique_shape_points()
        if not points:
            return func(*args, **kwargs)

        center_x = sum(point.x for point in points) / len(points)
        center_y = sum(point.y for point in points) / len(points)
        center_z = sum(point.z for point in points) / len(points)

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
        for point in self._unique_shape_points():
            vec = np.array([point.x, point.y, point.z, 1])
            transformed_vec = vec @ matrix
            w = float(transformed_vec[3])
            if abs(w) < self._W_EPS:
                raise ZeroDivisionError("Homogeneous w is ~0 during transform; check your matrix/projection.")
            point.x = float(transformed_vec[0]) / w
            point.y = float(transformed_vec[1]) / w
            point.z = float(transformed_vec[2]) / w
        return self
    
    def projection(self, matrix: np.ndarray) -> Iterator[BasePoint]:
        for point in self._unique_shape_points():
            vec = np.array([point.x, point.y, point.z, 1])
            transformed_vec = vec @ matrix
            w = float(transformed_vec[3])
            if abs(w) < self._W_EPS:
                raise ZeroDivisionError("Homogeneous w is ~0 during projection; check your matrix/projection.")
            yield BasePoint(
                float(transformed_vec[0]) / w,
                float(transformed_vec[1]) / w,
                float(transformed_vec[2]) / w,
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
        new = copy.copy(self)
        if isinstance(other, BaseEdge):
            new.points = self.points + other.points
        else:
            new.points = self.points + [other]
        return new
        
    def __iadd__(self, other: Union[PointType, EdgeType]) -> Self:
        if isinstance(other, BaseEdge):
            self.points.extend(other.points)

        else:
            self.points.append(other)

        return self
    
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
        new = copy.copy(self)
        new.edges = self.edges + [other]
        return new
    
    def __iadd__(self, other: EdgeType) -> Self:
        self.edges.append(other)
        return self
    
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
        seen: set[int] = set()
        unique: List[PointType] = []
        for edge in self.edges:
            for point in edge.points:
                pid = id(point)
                if pid in seen:
                    continue
                seen.add(pid)
                unique.append(point)
        return unique
    
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


class BaseObject(metaclass=BaseShape):
    def __init__(
            self,
            points: Sequence[PointType] = (),
            edges: Sequence[EdgeType] = (),
            polygons: Sequence[PolygonType] = (),
        ) -> None:
        self.points: List[PointType] = list(points)
        self.edges: List[EdgeType] = list(edges)
        self.polygons: List[PolygonType] = list(polygons)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(points={self.points}, "
            f"edges={self.edges}, polygons={self.polygons}, id={hex(id(self))})"
        )


class Object(BaseObject, ITransformable):
    def __init__(
            self,
            points: Sequence[PointType] = (),
            edges: Sequence[EdgeType] = (),
            polygons: Sequence[PolygonType] = (),
        ) -> None:
        BaseObject.__init__(self, points, edges, polygons)
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
            return (
                f"{self.__class__.__name__}(points={self.points}, "
                f"edges={self.edges}, polygons={self.polygons}, "
                f"{features}, id={hex(id(self))})"
            )
        
        else:
            return (
                f"{self.__class__.__name__}(points={self.points}, "
                f"edges={self.edges}, polygons={self.polygons}, "
                f"id={hex(id(self))})"
            )


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
        self.objects: List[ObjectType] = []
        ITransformable.__init__(self)

    @override
    def get_shape_points(self) -> List[PointType]:
        return self.points

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(width={self.width}, height={self.height}, "
            f"objects={self.objects}, id={hex(id(self))})"
        )
    
    def __add__(self, obj: Shape | ObjectType) -> Self:
        new = copy.copy(self)
        new.points = list(self.points)
        new.edges = list(self.edges)
        new.polygons = list(self.polygons)
        new.objects = list(self.objects)
        new += obj
        return new
    
    def __iadd__(self, obj: Shape | ObjectType) -> Self:
        if (isinstance(point := obj, BasePoint)):
            if point not in self.points:
                self.points.append(point)
        elif isinstance(edge := obj, BaseEdge):
            for point in edge:
                self.__iadd__(point)
            if edge not in self.edges:
                self.edges.append(edge)
        elif isinstance(polygon := obj, BasePolygon):
            for edge in polygon:
                self.__iadd__(edge)
            if polygon not in self.polygons:
                self.polygons.append(polygon)
        elif isinstance(obj := obj, BaseObject):
            for point in obj.points:
                self.__iadd__(point)
            for edge in obj.edges:
                self.__iadd__(edge)
            for polygon in obj.polygons:
                self.__iadd__(polygon)
            if obj not in self.objects:
                self.objects.append(obj)

        return self
    
    def __sub__(self, obj: Shape | ObjectType) -> Self:
        new = copy.copy(self)
        new.points = list(self.points)
        new.edges = list(self.edges)
        new.polygons = list(self.polygons)
        new.objects = list(self.objects)
        new -= obj
        return new
    
    def __isub__(self, obj: Shape | ObjectType) -> Self:
        if isinstance(obj, BasePoint):
            self.points.remove(obj)
        elif isinstance(obj, BaseEdge):
            self.edges.remove(obj)
        elif isinstance(obj, BasePolygon):
            self.polygons.remove(obj)
        else:
            self.objects.remove(obj)

        return self
    
    def clear(self) -> None:
        self.points.clear()
        self.edges.clear()
        self.polygons.clear()
        self.objects.clear()

class Vertex(Point):
    def __init__(self, x: Number, y: Number, z: Number, w: Number = 1.0, u : Number = 0.0, v: Number = 0.0, normal: tuple[Number, Number, Number] | None = None) -> None:
        super().__init__(x, y, z, w)
        self.u = u
        self.v = v
        self.normal = normal

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.x}, {self.y}, {self.z}, "
            f"w={self.w}, uv=({self.u}, {self.v}), normal={self.normal}, id={hex(id(self))})"
        )