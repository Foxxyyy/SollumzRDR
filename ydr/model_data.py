"""Reads DrawableModel mesh data into numpy arrays."""
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple, Tuple

from ..tools.drawablehelper import get_model_xmls_by_lod
from ..sollumz_properties import LODLevel, SollumzGame
from ..cwxml.drawable import Bone, Drawable, DrawableModel, Geometry, LodList, VertexBuffer
from ..cwxml.drawable_RDR import VERT_ATTR_DTYPES

current_game = SollumzGame.GTA

class MeshData(NamedTuple):
    vert_arr: NDArray
    ind_arr: NDArray[np.uint]
    mat_inds: NDArray[np.uint]


class ModelData(NamedTuple):
    mesh_data_lods: dict[LODLevel, MeshData]
    # Used for storing drawable model properties
    xml_lods: dict[LODLevel, DrawableModel]
    bone_index: int


def get_model_data(drawable_xml: Drawable) -> list[ModelData]:
    """Get ModelData for each DrawableModel."""
    global current_game
    current_game = drawable_xml.game
    model_datas: list[ModelData] = []
    model_xmls, bone_inds = get_lod_model_xmls(drawable_xml)
    for model_lods, bone_ind in zip(model_xmls, bone_inds):
        model_data = ModelData(
            mesh_data_lods={
                lod_level: mesh_data_from_xml(model_xml) for lod_level, model_xml in model_lods.items()
            },
            xml_lods={
                lod_level: model_xml for lod_level, model_xml in model_lods.items()
            },
            bone_index=bone_ind
        )

        model_datas.append(model_data)

    return model_datas


def get_model_data_split_by_group(drawable_xml: Drawable) -> list[ModelData]:
    model_datas = get_model_data(drawable_xml)

    return [split_data for model_data in model_datas for split_data in split_model_by_group(model_data, drawable_xml.skeleton.bones)]


def split_model_by_group(model_data: ModelData, bones: list[Bone]) -> list[ModelData]:
    """Split ModelData by vertex group"""
    model_datas: list[ModelData] = []
    mesh_data_by_bone: dict[int, dict[LODLevel, MeshData]] = defaultdict(dict)

    for lod_level, mesh_data in model_data.mesh_data_lods.items():
        is_skinned = "BlendWeights" in mesh_data.vert_arr.dtype.names

        if not is_skinned:
            mesh_data_by_bone[model_data.bone_index][lod_level] = mesh_data
            continue

        for i, face_inds in get_group_face_inds(mesh_data, bones).items():
            vert_arr, ind_arr = get_faces_subset(
                mesh_data.vert_arr, mesh_data.ind_arr, face_inds)

            group_mesh_data = MeshData(
                vert_arr,
                ind_arr,
                mesh_data.mat_inds[face_inds]
            )

            mesh_data_by_bone[i][lod_level] = group_mesh_data

    for i, mesh_data_lods in mesh_data_by_bone.items():
        model_datas.append(ModelData(
            mesh_data_lods=mesh_data_lods,
            xml_lods=model_data.xml_lods,
            bone_index=i
        ))

    return model_datas


def get_group_face_inds(mesh_data: MeshData, bones: list[Bone]):
    """Get face indices split by vertex group. Overlapping vertex groups are merged
    based on bone parenting."""
    group_inds: dict[int, set] = defaultdict(list)

    blend_inds = mesh_data.vert_arr["BlendIndices"]
    weights = mesh_data.vert_arr["BlendWeights"]

    num_tris = int(len(mesh_data.ind_arr) / 3)
    faces = mesh_data.ind_arr.reshape((num_tris, 3))

    # Get all the BlendIndices and BlendWeights in each face
    face_blend_inds = blend_inds[faces]
    face_weights = weights[faces]
    # Any given face could be in a maximum of 12 vertex groups (3 verts * 4 possible groups per vert)
    face_blend_inds = face_blend_inds.reshape((num_tris, 12))
    face_weights = face_blend_inds.reshape((num_tris, 12))

    # Mapping of blend indices in each face where (BlendIndex, BlendWeight) pairs are not (0, 0)
    blend_inds_mask = np.logical_or(face_blend_inds != 0, face_weights != 0)
    # Maps group indices to the group index of the object they should be parented to
    parent_map = get_group_parent_map(face_blend_inds, bones)

    for i, all_blend_inds in enumerate(face_blend_inds):
        # Valid BlendIndices are those where either the index or weight is not 0
        valid_blend_inds = all_blend_inds[blend_inds_mask[i]]

        if valid_blend_inds.size == 0:
            group_ind = 0
        else:
            group_ind = parent_map[valid_blend_inds[0]]

        group_inds[group_ind].append(i)

    return {i: np.array(face_inds, dtype=np.uint32) for i, face_inds in group_inds.items()}


def get_group_parent_map(face_blend_inds: NDArray[np.uint32], bones: list[Bone]) -> dict[int, set]:
    """Get a mapping of each blend index to the blend index of the object they should be parented to."""
    # Mapping of each blend index to blend indices with overlapping faces
    group_relations: dict[int, set[int]] = defaultdict(set)
    parent_map: dict[int, int] = {}
    group_inds = np.unique(face_blend_inds)

    for group_ind in group_inds:
        occurences = np.any(face_blend_inds == group_ind, axis=1)
        related_groups = np.unique(face_blend_inds[occurences].flatten())
        # Ignore 0 group because all vertex groups are a part of group 0
        group_relations[group_ind] = [
            i for i in related_groups if i != 0 and i != group_ind]

    for blend_ind, blend_inds in group_relations.items():
        # blend_ind does not overlap with any other vertex groups, so it can be created as its own object
        if not blend_inds:
            parent_map[blend_ind] = blend_ind
            continue

        # Find a parent bone that is shared between all blend_inds. All faces with blend_inds vertex groups will be
        # created as a single object
        parent_map[blend_ind] = find_common_bone_parent(blend_inds, bones)

    return parent_map


def find_common_bone_parent(bone_inds: list[int], bones: list[Bone]) -> int:
    bone_parents = [get_all_bone_parents(
        i, bones) for i in bone_inds if bones[i].parent_index > 0]

    if not bone_parents:
        return 0

    common_bones = bone_parents[0]

    for array in bone_parents[1:]:
        common_bones = np.intersect1d(common_bones, array)

    if len(common_bones) == 0:
        return 0

    # Get the parent thats highest in the bone hierarchy
    return np.min(common_bones)


def get_all_bone_parents(bone_ind: int, bones: list[Bone]):
    parent_inds = []
    current_bone_ind = bone_ind

    while bones[current_bone_ind].parent_index > 0:
        current_bone_ind = bones[current_bone_ind].parent_index
        parent_inds.append(current_bone_ind)

    return parent_inds


def get_faces_subset(vert_arr: NDArray, ind_arr: NDArray[np.uint32], face_inds: NDArray[np.uint32]) -> MeshData:
    """Get subset of vertex array and index array by face."""
    # First get valid faces in the subset in case the new subset of indices is not divisble by 3.
    num_tris = int(len(ind_arr) / 3)
    faces = ind_arr.reshape((num_tris, 3))

    subset_inds = faces[face_inds].flatten()

    # Map old vert inds to new vert inds
    # TODO: Remap inds using numpy?
    vert_inds_map: dict[int, int] = {}
    vert_inds = []
    new_inds = []

    for vert_ind in subset_inds:
        if vert_ind in vert_inds_map:
            new_inds.append(vert_inds_map[vert_ind])
        else:
            new_vert_ind = len(vert_inds_map)
            new_inds.append(new_vert_ind)
            vert_inds_map[vert_ind] = new_vert_ind
            vert_inds.append(vert_ind)

    new_vert_arr = vert_arr[vert_inds]
    new_ind_arr = np.array(new_inds, dtype=np.uint32)

    return new_vert_arr, new_ind_arr


def get_lod_model_xmls(drawable_xml: Drawable) -> Tuple[list[dict[LODLevel, DrawableModel]], list[int]]:
    """Gets mapping of LOD levels for each DrawableModel. Also returns a list of bone indices for each model."""
    model_xmls: dict[int, dict[LODLevel, DrawableModel]] = defaultdict(dict)
    bone_inds: list[int] = []

    for lod_level, models in get_model_xmls_by_lod(drawable_xml).items():
        if current_game == SollumzGame.RDR and isinstance(models, LodList):
            models = models.models
        if models is None:
            print("Skipping LOD level since lod model returned None")
            continue
        for i, model_xml in enumerate(models):
            if i not in model_xmls:
                # Each corresponding DrawableModel will always have the same bone index across all LODs (verified with CodeWalker)
                bone_inds.append(model_xml.bone_index)

            model_xmls[i][lod_level] = model_xml

    return list(model_xmls.values()), bone_inds


def mesh_data_from_xml(model_xml: DrawableModel) -> MeshData:
    geoms = get_valid_geoms(model_xml)

    return MeshData(
        ind_arr=get_model_joined_ind_arr(geoms),
        vert_arr=get_model_joined_vert_arr(geoms),
        mat_inds=get_model_poly_mat_inds(geoms)
    )


def get_model_joined_ind_arr(geoms: list[Geometry]) -> NDArray[np.uint32]:
    """Get joined indices array for the model."""
    ind_arrs: list[NDArray[np.uint32]] = []
    num_verts = 0

    for geom in geoms:
        ind_arr = None
        if current_game == SollumzGame.RDR:
            ind_arr = geom.indices
        elif current_game == SollumzGame.GTA:
            ind_arr = geom.index_buffer.data

        if num_verts > 0:
            ind_arr += num_verts

        ind_arrs.append(ind_arr)
        if current_game == SollumzGame.RDR:
            num_verts += len(geom.vertices)
        elif current_game == SollumzGame.GTA:
            num_verts += len(geom.vertex_buffer.data)

    return np.concatenate(ind_arrs)


def get_model_joined_vert_arr(geoms: list[Geometry]) -> NDArray:
    arr_dtype = get_model_vert_buffer_dtype(geoms)
    vert_arrs: list[NDArray] = []

    for geom in geoms:
        if current_game == SollumzGame.RDR:
            vert_arr = geom.vertices
        elif current_game == SollumzGame.GTA:
            vert_arr = geom.vertex_buffer.data

        if vert_arr is None:
            continue

        if geom.bone_ids:
            apply_bone_ids(vert_arr, np.array(geom.bone_ids))

        geom_vert_arr = np.zeros(len(vert_arr), dtype=arr_dtype)

        for name in vert_arr.dtype.names:
            geom_vert_arr[name] = vert_arr[name]

        vert_arrs.append(geom_vert_arr)

    return np.concatenate(vert_arrs)


def get_model_vert_buffer_dtype(geoms: list[Geometry]) -> np.dtype:
    """Get the dtype of the structured array of the joined geometry vertex buffers."""
    arr_dtype = []    
    if current_game == SollumzGame.RDR:
        merged_dtype_names = []
        for geom in geoms:
            geo_dtype = geom.vertices.dtype
            for dtype in geo_dtype.names:
                for attr_name, attr_dtype in VERT_ATTR_DTYPES.items():
                    if attr_dtype[0] in dtype:
                        x = list(attr_dtype.copy())
                        x[0] = dtype
                        if dtype not in merged_dtype_names:
                            arr_dtype.append(tuple(x))
                            merged_dtype_names.append(dtype)
        del merged_dtype_names[:]; del merged_dtype_names
    else:
        used_attrs: set[Tuple] = set(
            name for geom in geoms for name in geom.vertex_buffer.data.dtype.names)
        for attr_name, attr_dtype in VertexBuffer.VERT_ATTR_DTYPES.items():
            if attr_name not in used_attrs:
                continue

            arr_dtype.append(attr_dtype)

    return np.dtype(arr_dtype)


def apply_bone_ids(vert_arr: NDArray, bone_ids: NDArray[np.uint32]):
    if "BlendIndices" not in vert_arr.dtype.names:
        return vert_arr

    vert_arr["BlendIndices"] = bone_ids[vert_arr["BlendIndices"]]


def get_model_poly_mat_inds(geoms: list[Geometry]):
    mat_ind_arrs = []

    for geom in geoms:
        if current_game == SollumzGame.GTA:
            num_tris = round(len(geom.index_buffer.data) / 3)
            mat_inds = np.full((num_tris,), geom.shader_index)
        elif current_game == SollumzGame.RDR:
            num_tris = round(len(geom.indices) / 3)
            mat_inds = np.full((num_tris,), geom.shader_index)

        mat_ind_arrs.append(mat_inds)

    return np.concatenate(mat_ind_arrs)


def get_valid_geoms(model_xml: DrawableModel) -> list[Geometry]:
    """Get geometries with mesh data in model_xml."""
    if current_game == SollumzGame.GTA:
        return [geom for geom in model_xml.geometries if geom.vertex_buffer.data is not None and geom.index_buffer.data is not None]
    elif current_game == SollumzGame.RDR:
        return [geom for geom in model_xml.geometries if geom.vertices is not None and geom.indices is not None]
