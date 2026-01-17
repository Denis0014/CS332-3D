from dataclasses import dataclass
import numpy as np
from typing import Optional, Tuple

from geometry import PointType, Vertex


@dataclass
class VertexOutput:
    def __init__(self, position: np.ndarray, normal: Optional[np.ndarray], uv: Tuple[float, float]) -> None:
        self.position = position
        self.normal = normal
        self.uv = uv


@dataclass
class PointLight:
    def __init__(
            self,
            position: Tuple[float, float, float],
            ambient: Tuple[float, float, float],
            diffuse: Tuple[float, float, float],
            specular: Tuple[float, float, float],
            attenuation: Tuple[float, float, float]) -> None:
        self.position = np.array(position, dtype=float)
        self.ambient = np.array(ambient, dtype=float)
        self.diffuse = np.array(diffuse, dtype=float)
        self.specular = np.array(specular, dtype=float)
        self.attenuation = attenuation  # (constant, linear, quadratic)


@dataclass
class Material:
    def __init__(
            self,
            ambient: Tuple[float, float, float],
            diffuse: Tuple[float, float, float],
            specular: Tuple[float, float, float],
            shininess: float) -> None:
        self.ambient = np.array(ambient, dtype=float)
        self.diffuse = np.array(diffuse, dtype=float)
        self.specular = np.array(specular, dtype=float)
        self.shininess = shininess


class VertexShader:
    def __init__(
        self,
        model_matrix: Optional[np.ndarray] = None,
        view_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.model_matrix = model_matrix if model_matrix is not None else np.eye(4)
        self.view_matrix = view_matrix if view_matrix is not None else np.eye(4)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    def __call__(self, vertex: PointType) -> VertexOutput:
        w = getattr(vertex, "w", 1.0)
        position = np.array([vertex.x, vertex.y, vertex.z, w], dtype=float)
        world = position @ self.model_matrix @ self.view_matrix

        normal: Optional[np.ndarray] = None
        if isinstance(vertex, Vertex) and vertex.normal is not None:
            normal_vec = np.array(vertex.normal, dtype=float)
            try:
                normal_matrix = np.linalg.inv(self.model_matrix[:3, :3]).T
                normal = self._normalize(normal_vec @ normal_matrix)
            except np.linalg.LinAlgError:
                normal = self._normalize(normal_vec)

        uv = (
            getattr(vertex, "u", 0.0),
            getattr(vertex, "v", 0.0),
        )

        return VertexOutput(world, normal, uv)


class FragmentShader:
    def __init__(
        self,
        light: Optional[PointLight] = None,
        material: Optional[Material] = None,
        view_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.light = light or PointLight(
            position=(0.0, 2.0, 0.0),
            ambient=(0.1, 0.1, 0.1),
            diffuse=(1.0, 1.0, 1.0),
            specular=(1.0, 1.0, 1.0),
            attenuation=(1.0, 0.05, 0.01),
        )
        self.material = material or Material(
            ambient=(0.6, 0.6, 0.6),
            diffuse=(0.6, 0.6, 0.6),
            specular=(0.6, 0.6, 0.6),
            shininess=16.0,
        )
        self.view_pos = np.array(view_pos, dtype=float)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm

    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        r, g, b = rgb
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    def shade(
        self,
        position: Optional[np.ndarray],
        normal: Optional[np.ndarray],
        material: Optional[Material] = None,
    ) -> Tuple[int, int, int]:
        mat = material or self.material

        if normal is None or np.linalg.norm(normal) == 0.0:
            ambient = mat.ambient * self.light.ambient
            color = np.clip(ambient, 0.0, 1.0)
            r, g, b = (int(channel * 255) for channel in color)
            return (r, g, b)

        n = self._normalize(normal)
        if position is None:
            pos = np.zeros(3, dtype=float)
        else:
            pos = np.array(position, dtype=float)[:3]

        light_vec = self.light.position - pos
        distance = float(np.linalg.norm(light_vec))
        if distance == 0.0:
            light_dir = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            light_dir = light_vec / distance

        view_dir = self._normalize(self.view_pos - pos)
        reflect_dir = 2.0 * np.dot(n, light_dir) * n - light_dir

        diff = max(float(np.dot(n, light_dir)), 0.0)
        spec = max(float(np.dot(view_dir, reflect_dir)), 0.0) ** mat.shininess

        attenuation = 1.0 / (
            self.light.attenuation[0]
            + self.light.attenuation[1] * distance
            + self.light.attenuation[2] * distance * distance
        )

        ambient = mat.ambient * self.light.ambient
        diffuse = mat.diffuse * self.light.diffuse * diff
        specular = mat.specular * self.light.specular * spec

        color = ambient + (diffuse + specular) * attenuation
        color = np.clip(color, 0.0, 1.0)
        r, g, b = (int(channel * 255) for channel in color)
        return (r, g, b)
