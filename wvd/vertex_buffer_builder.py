import bpy
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, Optional

from ..tools.meshhelper import flip_uvs
from ..cwxml.drawable import VertexBuffer
from ..cwxml.drawable_RDR import VERT_ATTR_DTYPES

def get_bone_by_vgroup(vgroups: bpy.types.VertexGroups, bones: list[bpy.types.Bone]):
    bone_ind_by_name: dict[str, int] = {
        b.name: i for i, b in enumerate(bones)}

    return {i: bone_ind_by_name[group.name] if group.name in bone_ind_by_name else 0 for i, group in enumerate(vgroups)}


def remove_arr_field(name: str, vertex_arr: NDArray):
    names = [n for n in vertex_arr.dtype.names if n != name]
    return vertex_arr[names]


def remove_unused_colors(vertex_arr: NDArray, used_colors: set[str]) -> NDArray:
    """Remove color layers that aren't used by the shader"""
    new_names = []

    for name in vertex_arr.dtype.names:
        if "Colour" in name and name not in used_colors:
            continue
        new_names.append(name)

    return vertex_arr[new_names]


def remove_unused_uvs(vertex_arr: NDArray, used_texcoords: set[str]) -> NDArray:
    """Remove UV layers that aren't used by the shader"""
    new_names = []

    for name in vertex_arr.dtype.names:
        if "TexCoord" in name and name not in used_texcoords:
            continue
        new_names.append(name)

    return vertex_arr[new_names]


def round_array(arr: NDArray, decimals: int = 5) -> NDArray:
    """Round a numpy array to a certain number of decimal places."""
    return np.round(arr, decimals=decimals)


def dedupe_and_get_indices(vertex_arr: NDArray) -> Tuple[NDArray, NDArray[np.uint32]]:
    """Remove duplicate vertices based on available attributes in the vertex buffer and get new vertex indices."""
    index_map: Dict[tuple, int] = {}
    unique_vertices: list = []
    indices: list = []

    # Dynamically check for the existence of attributes
    has_normal = 'Normal' in vertex_arr.dtype.names
    has_color0 = 'Colour0' in vertex_arr.dtype.names
    has_color1 = 'Colour1' in vertex_arr.dtype.names

    for i, vert in enumerate(vertex_arr):
        # Always include position in the key (rounded to avoid floating-point issues)
        vert_key = (tuple(vert['Position']),)

        # Conditionally add UVs, colors, and normals to the key (rounded)
        if has_color0:
            vert_key += (tuple(vert['Colour0']),)
        if has_color1:
            vert_key += (tuple(vert['Colour1']),)
        if has_normal:
            vert_key += (tuple(round_array(vert['Normal'], decimals=5)),)

        # Check if this vertex combination already exists
        if vert_key not in index_map:
            index_map[vert_key] = len(unique_vertices)
            unique_vertices.append(vert)
        else:
            print(f"Duplicate found for vertex {i}: {vert_key}")

        indices.append(index_map[vert_key])

    print(f"Original vertex count: {len(vertex_arr)}")
    print(f"Unique vertex count: {len(unique_vertices)}")
    
    return np.array(unique_vertices), np.array(indices, dtype=np.uint32)

def get_indices_without_deduplication(vertex_arr: NDArray) -> Tuple[NDArray, NDArray[np.uint32]]:
    """Return the original vertex array and an array of indices ranging from 0 to the length of the vertex array."""
    indices = np.arange(len(vertex_arr), dtype=np.uint32)
    return vertex_arr, indices

class VertexBufferBuilder:
    """Builds Geometry vertex buffers from a mesh."""

    def __init__(self, mesh: bpy.types.Mesh, bone_by_vgroup: Optional[dict[int, int]] = None):
        self.mesh = mesh

        self._bone_by_vgroup = bone_by_vgroup
        self._has_weights = bone_by_vgroup is not None

        vert_inds = np.empty(len(mesh.loops), dtype=np.uint32)
        self.mesh.loops.foreach_get("vertex_index", vert_inds)

        self._vert_inds = vert_inds

    def build(self):
        if not self.mesh.loop_triangles:
            self.mesh.calc_loop_triangles()

        self.mesh.calc_normals_split()

        mesh_attrs = self._collect_attrs()
        return self._structured_array_from_attrs(mesh_attrs)

    def _collect_attrs(self):
        """Returns a dict mapping arrays of all GTAV vertex attributes in ``self.mesh`` stored on the loop domain."""
        mesh_attrs = {}

        mesh_attrs["Position"] = self._get_positions()

        if self._has_weights:
            data = self._get_weights_indices()

            mesh_attrs["BlendWeights"] = data[0]
            mesh_attrs["BlendIndices"] = data[1]
            mesh_attrs["BlendWeights2"] = data[2]
            mesh_attrs["BlendIndices2"] = data[3]

        mesh_attrs["Normal"] = self._get_normals()
        mesh_attrs["Tangent"] = self._get_tangents()
        mesh_attrs["Tangent1"] = mesh_attrs["Tangent"].copy()
        mesh_attrs["Tangent2"] = mesh_attrs["Tangent"].copy()

        colors = self._get_colors()

        for i, color in enumerate(colors):
            mesh_attrs[f"Colour{i}"] = color

        uvs = self._get_uvs()

        for i, uv in enumerate(uvs):
            mesh_attrs[f"TexCoord{i}"] = uv

        mesh_attrs["Tangent"] = self._get_tangents()

        return mesh_attrs

    def _structured_array_from_attrs(self, mesh_attrs: dict[str, NDArray]):
        """Combine ``mesh_attrs`` into single structured array."""
        # Data type for vertex data structured array
        struct_dtype = []
        for attr_name in mesh_attrs:
            for val_list in VERT_ATTR_DTYPES.values():
                if val_list[0] in attr_name:
                    item = val_list.copy()
                    item[0] = attr_name
                    item = tuple(item)
                    struct_dtype.append(item)
                    break

        vertex_arr = np.empty(len(self._vert_inds), dtype=struct_dtype)

        for attr_name, arr in mesh_attrs.items():
            vertex_arr[attr_name] = arr

        return vertex_arr

    def _get_positions(self):
        positions = np.empty(len(self.mesh.vertices) * 3, dtype=np.float32)
        self.mesh.attributes["position"].data.foreach_get("vector", positions)
        positions = np.reshape(positions, (len(self.mesh.vertices), 3))

        return positions[self._vert_inds]

    def _get_normals(self):
        normals = np.empty(len(self.mesh.loops) * 3, dtype=np.float32)
        self.mesh.loops.foreach_get("normal", normals)
        
        processed_normal = np.zeros((len(self.mesh.loops), 4), dtype=np.float32)
        processed_normal[:, :3] = np.reshape(normals, (len(self.mesh.loops), 3))
        condition = processed_normal[:, 2] < 0
        processed_normal[:, 3] = np.where(condition, -1, 0)
        return processed_normal

    def _get_weights_indices(self) -> Tuple[NDArray[np.uint32], NDArray[np.uint32]]:
        """Get all BlendWeights and BlendIndices."""
        num_verts = len(self.mesh.vertices)
        bone_by_vgroup = self._bone_by_vgroup

        ind_arr = np.zeros((num_verts, 4), dtype=np.uint32)
        weights_arr = np.zeros((num_verts, 4), dtype=np.float32)

        ind_arr2 = np.zeros((num_verts, 4), dtype=np.uint32)
        weights_arr2 = np.zeros((num_verts, 4), dtype=np.float32)

        for i, vert in enumerate(self.mesh.vertices):
            for j, grp in enumerate(vert.groups):
                if j < 4:
                    weights_arr[i][j] = grp.weight
                    ind_arr[i][j] = bone_by_vgroup[grp.group]
                elif j >= 4 and j < 8:
                    weights_arr2[i][j] = grp.weight
                    ind_arr2[i][j] = bone_by_vgroup[grp.group]
                else:
                    break

        weights_arr = self._normalize_weights(weights_arr)
        weights_arr2 = self._normalize_weights(weights_arr2)

        weights_arr, ind_arr = self._sort_weights_inds(weights_arr, ind_arr)
        weights_arr2, ind_arr2 = self._sort_weights_inds(weights_arr2, ind_arr2)

        weights_arr = self._convert_to_int_range(weights_arr)
        weights_arr2 = self._convert_to_int_range(weights_arr2)

        # Return on loop domain
        return [weights_arr[self._vert_inds], ind_arr[self._vert_inds], weights_arr2[self._vert_inds], ind_arr2[self._vert_inds]]

    def _sort_weights_inds(self, weights_arr: NDArray[np.float32], ind_arr: NDArray[np.uint32]):
        """Sort BlendWeights and BlendIndices."""
        # Blend weights and indices are sorted by weights in ascending order starting from the 3rd index and continues to the left
        # Why? I dont know :/
        sort_inds = np.argsort(weights_arr, axis=1)

        # Apply sort on axis 1
        weights_sorted = np.take_along_axis(weights_arr, sort_inds, axis=1)
        ind_sorted = np.take_along_axis(ind_arr, sort_inds, axis=1)

        # Return with index shifted by 3
        return np.roll(weights_sorted, 3, axis=1), np.roll(ind_sorted, 3, axis=1)

    def _normalize_weights(self, weights_arr: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize weights such that their sum is 1."""
        row_sums = weights_arr.sum(axis=1, keepdims=True)
        return np.divide(weights_arr, row_sums, out=np.zeros_like(
            weights_arr), where=row_sums != 0)

    def _convert_to_int_range(self, arr: NDArray[np.float32]) -> NDArray[np.uint32]:
        """Convert float array from range 0-1 to range 0-255"""
        return (np.rint(arr * 255)).astype(np.uint32)

    def _get_colors(self) -> list[NDArray[np.uint32]]:
        num_loops = len(self.mesh.loops)

        def _is_valid_color_attr(attr: bpy.types.Attribute):
            return (attr.domain == "CORNER" and
                    # `TintColor` only used for the tint shaders/geometry nodes
                    not attr.name.startswith("TintColor") and
                    # Name prefixed by `.` indicate a reserved attribute name for Blender
                    # e.g. `.a_1234` for anonymous attributes
                    # https://projects.blender.org/blender/blender/issues/97452
                    not attr.name.startswith("."))

        color_attrs = [attr for attr in self.mesh.color_attributes if _is_valid_color_attr(attr)]

        # Always have at least 1 color layer
        if len(color_attrs) == 0:
            return [np.full((len(self._vert_inds), 4), 255, dtype=np.uint32)]

        color_layers = []

        for color_attr in color_attrs:
            colors = np.empty(num_loops * 4, dtype=np.float32)
            color_attr.data.foreach_get("color_srgb", colors)

            colors = self._convert_to_int_range(colors)

            color_layers.append(np.reshape(colors, (num_loops, 4)))

        return color_layers

    def _get_uvs(self) -> list[NDArray[np.float32]]:
        num_loops = len(self.mesh.loops)
        # UV mesh attributes (maximum of 8 for GTAV shaders)
        uv_attrs = [attr for attr in self.mesh.attributes if attr.data_type ==
                    'FLOAT2' and attr.domain == 'CORNER'][:8]
        uv_layers: list[NDArray[np.float32]] = []

        for uv_attr in uv_attrs:
            uvs = np.empty(num_loops * 2, dtype=np.float32)
            uv_attr.data.foreach_get("vector", uvs)
            uvs = np.reshape(uvs, (num_loops, 2))

            flip_uvs(uvs)

            uv_layers.append(uvs)

        return uv_layers

    def _get_tangents(self):
        mesh = self.mesh
        num_loops = len(mesh.loops)

        if not mesh.uv_layers:
            return np.zeros((num_loops, 4), dtype=np.float32)

        mesh.calc_tangents()

        tangents = np.empty(num_loops * 3, dtype=np.float32)
        bitangent_signs = np.empty(num_loops, dtype=np.float32)

        mesh.loops.foreach_get("tangent", tangents)
        mesh.loops.foreach_get("bitangent_sign", bitangent_signs)

        tangents = np.reshape(tangents, (num_loops, 3))
        bitangent_signs = np.reshape(bitangent_signs, (-1, 1))

        return np.concatenate((tangents, bitangent_signs), axis=1)
