import xml.etree.ElementTree as ET
import os
from abc import ABC, abstractmethod
from .drawable_RDR import VERT_ATTR_DTYPES

from ..sollumz_properties import SollumzGame
from .element import (
    ElementTree,
    ListProperty,
    TextProperty,
    AttributeProperty,
)
from .drawable import VertexLayoutList
from ..tools import jenkhash
from typing import Optional
from enum import Enum


current_game = SollumzGame.RDR1

class FileNameList(ListProperty):
    class FileName(TextProperty):
        tag_name = "Item"

    list_type = FileName
    tag_name = "FileName"


class LayoutList(ListProperty):
    class Layout(VertexLayoutList):
        tag_name = "Item"

    list_type = Layout
    tag_name = "Layout"


class ShaderParameterType(str, Enum):
    TEXTURE = "Texture"
    FLOAT = "float"
    FLOAT2 = "float2"
    FLOAT3 = "float3"
    FLOAT4 = "float4"
    FLOAT4X4 = "float4x4"
    SAMPLER = "Sampler"
    CBUFFER = "CBuffer"
    UNKNOWN = "Unknown"


class ShaderParameterSubtype(str, Enum):
    RGB = "rgb"
    RGBA = "rgba"
    BOOL = "bool"


class ShaderParameterDef(ElementTree, ABC):
    tag_name = "Item"

    @property
    @abstractmethod
    def type() -> ShaderParameterType:
        raise NotImplementedError

    def __init__(self):
        super().__init__()
        self.name = AttributeProperty("name")
        self.type = AttributeProperty("type", self.type)
        self.hidden = AttributeProperty("hidden", False)


class ShaderParameterTextureDef(ShaderParameterDef):
    type = ShaderParameterType.TEXTURE

    def __init__(self):
        super().__init__()
        self.uv = AttributeProperty("uv")


class ShaderParameterFloatVectorDef(ShaderParameterDef, ABC):
    def __init__(self):
        super().__init__()
        self.count = AttributeProperty("count", 0)

    @property
    def is_array(self):
        return self.count > 0


class ShaderParameterFloatDef(ShaderParameterFloatVectorDef):
    type = ShaderParameterType.FLOAT

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("x", 0.0)


class ShaderParameterFloat2Def(ShaderParameterFloatVectorDef):
    type = ShaderParameterType.FLOAT2

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("x", 0.0)
        self.y = AttributeProperty("y", 0.0)


class ShaderParameterFloat3Def(ShaderParameterFloatVectorDef):
    type = ShaderParameterType.FLOAT3

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("x", 0.0)
        self.y = AttributeProperty("y", 0.0)
        self.z = AttributeProperty("z", 0.0)


class ShaderParameterFloat4Def(ShaderParameterFloatVectorDef):
    type = ShaderParameterType.FLOAT4

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("x", 0.0)
        self.y = AttributeProperty("y", 0.0)
        self.z = AttributeProperty("z", 0.0)
        self.w = AttributeProperty("w", 0.0)


class ShaderParameterFloat4x4Def(ShaderParameterDef):
    type = ShaderParameterType.FLOAT4X4

    def __init__(self):
        super().__init__()


class ShaderParameterSamplerDef(ShaderParameterDef):
    type = ShaderParameterType.SAMPLER

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("sampler", 0)
        self.index = AttributeProperty("index", 0)


class ShaderParameterCBufferDef(ShaderParameterDef):
    type = ShaderParameterType.CBUFFER

    def __init__(self):
        super().__init__()
        self.buffer = AttributeProperty("buffer", 0)
        self.length = AttributeProperty("length", 0)
        self.offset = AttributeProperty("offset", 0)
        self.value_type = AttributeProperty("value_type")
        match self.value_type:
            case ShaderParameterType.FLOAT:
                self.x = AttributeProperty("x", 0.0)
            case ShaderParameterType.FLOAT2:
                self.x = AttributeProperty("x", 0.0)
                self.y = AttributeProperty("y", 0.0)
            case ShaderParameterType.FLOAT3:
                self.x = AttributeProperty("x", 0.0)
                self.y = AttributeProperty("y", 0.0)
                self.z = AttributeProperty("z", 0.0)
            case ShaderParameterType.FLOAT4:
                self.x = AttributeProperty("x", 0.0)
                self.y = AttributeProperty("y", 0.0)
                self.z = AttributeProperty("z", 0.0)
                self.z = AttributeProperty("w", 0.0)


class ShaderParameteUnknownDef(ShaderParameterDef):
    type = ShaderParameterType.UNKNOWN

    def __init__(self):
        super().__init__()


class ShaderParameterDefsList(ListProperty):
    list_type = ShaderParameterDef
    tag_name = "Parameters"

    @staticmethod
    def from_xml(element: ET.Element):
        new = ShaderParameterDefsList()
        for child in element.iter():
            if "type" in child.attrib:
                param_type = child.get("type")
                match param_type:
                    case ShaderParameterType.TEXTURE:
                        param = ShaderParameterTextureDef.from_xml(child)
                    case ShaderParameterType.FLOAT:
                        param = ShaderParameterFloatDef.from_xml(child)
                    case ShaderParameterType.FLOAT2:
                        param = ShaderParameterFloat2Def.from_xml(child)
                    case ShaderParameterType.FLOAT3:
                        param = ShaderParameterFloat3Def.from_xml(child)
                    case ShaderParameterType.FLOAT4:
                        param = ShaderParameterFloat4Def.from_xml(child)
                    case ShaderParameterType.FLOAT4X4:
                        param = ShaderParameterFloat4x4Def.from_xml(child)
                    case ShaderParameterType.SAMPLER:
                        param = ShaderParameterSamplerDef.from_xml(child)
                    case ShaderParameterType.CBUFFER:
                        param = ShaderParameterCBufferDef.from_xml(child)
                    case ShaderParameterType.UNKNOWN:
                        param = ShaderParameteUnknownDef.from_xml(child)
                    case _:
                        assert False, f"Unknown shader parameter type '{param_type}'"

                new.value.append(param)

        return new


class SemanticsList(ElementTree):
    tag_name = "Semantics"

    def __init__(self) -> None:
        super().__init__()
        self.values = []

    @staticmethod
    def from_xml(element: ET.Element):
        new = SemanticsList()
        for child in element.findall("Item"):
            new.values.append(child.text)
        return new


class ShaderDef(ElementTree):
    tag_name = "Item"

    render_bucket: int
    buffer_size: []
    uv_maps: dict[str, int]
    parameter_map: dict[str, ShaderParameterDef]

    def __init__(self):
        super().__init__()
        if current_game == SollumzGame.RDR1:
            self.filename = TextProperty("Name")
            self.render_bucket = 0
            self.buffer_size = []
            self.parameters = ShaderParameterDefsList("Params")
            self.semantics = SemanticsList()
            self.parameter_map = {}

    @property
    def required_tangent(self):
        if current_game == SollumzGame.RDR1:
            tangents = set()
            for semantic in self.semantics.values:
                current_semantic = None
                count = -1
                for this_semantic in semantic:
                    if current_semantic is None:
                        current_semantic = this_semantic
                    elif current_semantic != this_semantic:
                        current_semantic = this_semantic
                        count = -1
                    
                    entry = VERT_ATTR_DTYPES[this_semantic].copy()
                    
                    if count == -1 and entry[0] in ("Colour", "TexCoord"):
                        entry[0] = entry[0] + "0"
                    elif count >= 0:
                        entry[0] = entry[0] + str(count+1)
                    if "Tangent" in entry[0]:
                        tangents.add(entry[0])
                    count += 1
            return tangents
            

    @property
    def required_normal(self):
        if current_game == SollumzGame.RDR1:
            for semantic in self.semantics.values:
                for this_semantic in semantic:
                    entry = VERT_ATTR_DTYPES[this_semantic].copy()
                    if "Normal" in entry[0]:
                        return True
            return False

    @property
    def used_texcoords(self) -> set[str]:
        names = set()
        if current_game == SollumzGame.RDR1:
            for semantic in self.semantics.values:
                for this_semantic in semantic:
                    entry = VERT_ATTR_DTYPES[this_semantic].copy()
                    if "TexCoord" in entry[0]:
                        names.add(entry[0] + str(len(names)))
        return names

    @property
    def used_colors(self) -> set[str]:
        names = set()
        if current_game == SollumzGame.RDR1:
            for semantic in self.semantics.values:
                for this_semantic in semantic:
                    entry = VERT_ATTR_DTYPES[this_semantic].copy()
                    if "Colour" in entry[0]:
                        names.add(entry[0] + str(len(names)))
        return names

    @property
    def is_uv_animation_supported(self) -> bool:
        return "globalAnimUV0" in self.parameter_map and "globalAnimUV1" in self.parameter_map

    @classmethod
    def from_xml(cls, element: ET.Element) -> "ShaderDef":
        new: ShaderDef = super().from_xml(element)
        new.uv_maps = {
            p.name: p.uv for p in new.parameters if p.type == ShaderParameterType.TEXTURE and p.uv is not None
        }
        new.parameter_map = {p.name: p for p in new.parameters}
        return new


class ShaderManager:
    rdr1_shaderxml = os.path.join(os.path.dirname(__file__), "RDR1Shaders.xml")
    # Map shader filenames to base shader names
    _shaders_base_names: dict[ShaderDef, str] = {}
    _shaders: dict[str, ShaderDef] = {}
    _shaders_by_hash: dict[int, ShaderDef] = {}

    _rdr_shaders_base_names: dict[ShaderDef, str] = {}
    _rdr_shaders: dict[str, ShaderDef] = {}
    _rdr_shaders_by_hash: dict[int, ShaderDef] = {}

    _rdr1_shaders_base_names: dict[ShaderDef, str] = {}
    _rdr1_shaders: dict[str, ShaderDef] = {}
    _rdr1_shaders_by_hash: dict[int, ShaderDef] = {}

    blend_terrains = ["rdr2_terrain", "rdr2_terrain_blend"]
    cliff_road_terrains = ["rdr2_cliffwall_ao"]
    cliff_road_alpha_terrains = ["rdr2_cliffwall_alpha"]
    alphas = ["rdr2_alpha", "rdr2_alpha_bspec_ao_shareduv_character_hair", "rdr2_alpha_hair_a", "rdr2_shareduv_hair"]

    def alpha_shaders():
        return ShaderManager.alphas

    @staticmethod
    def load_shaders():
        global current_game
        rdr1tree = ET.parse(ShaderManager.rdr1_shaderxml)

        current_game = SollumzGame.RDR1
        for node in rdr1tree.getroot():
            base_name = node.find("Name").text

            filename_hash = jenkhash.Generate(base_name)
            render_bucket = node.find("DrawBucket").text.split(" ")
            if len(render_bucket) == 1:
                render_bucket = int(render_bucket[0])
            else:
                render_bucket = [int(x) for x in render_bucket]

            shader = ShaderDef.from_xml(node)
            shader.filename = base_name
            shader.render_bucket = render_bucket
            ShaderManager._rdr1_shaders[base_name] = shader
            ShaderManager._rdr1_shaders_by_hash[filename_hash] = shader
            ShaderManager._rdr1_shaders_base_names[shader] = base_name
        
        print("\Loaded total RDR1 shaders:", len(ShaderManager._rdr1_shaders))


    @staticmethod
    def find_shader(filename: str, game: SollumzGame = SollumzGame.RDR1) -> Optional[ShaderDef]:
        shader = None
        if game == SollumzGame.RDR1:
            shader = ShaderManager._rdr1_shaders.get(filename, None)
        if shader is None and filename.startswith("hash_"):
            filename_hash = int(filename[5:], 16)
            shader = ShaderManager._shaders_by_hash.get(filename_hash, None)
        return shader

    @staticmethod
    def find_shader_base_name(filename: str, game) -> Optional[str]:
        shader = ShaderManager.find_shader(filename, game)
        if shader is None:
            return None
        if game == SollumzGame.RDR1:
            return ShaderManager._rdr1_shaders_base_names[shader]

ShaderManager.load_shaders()
