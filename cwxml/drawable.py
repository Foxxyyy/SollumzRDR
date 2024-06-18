import io
import math
import os
from ..sollumz_properties import SollumzGame
from mathutils import Matrix, Vector
import numpy as np
from numpy.typing import NDArray
from ..tools.utils import np_arr_to_str
from typing import Optional
from abc import ABC as AbstractClass, abstractmethod
from xml.etree import ElementTree as ET
from .element import (
    AttributeProperty,
    FlagsProperty,
    Element,
    ColorProperty,
    ElementTree,
    ElementProperty,
    ListProperty,
    QuaternionProperty,
    TextProperty,
    ValueProperty,
    VectorProperty,
    Vector4Property,
    MatrixProperty
)
from collections.abc import MutableSequence
from .drawable_RDR import BoneMappingProperty, VertexLayout, VerticesProperty, IndicesProperty

current_game = SollumzGame.RDR1

class WVD:

    file_extension = ".wvd.xml"

    @staticmethod
    def from_xml_file(filepath):
        global current_game
        current_game = SollumzGame.RDR1
        return RDR1VisualDictionary.from_xml_file(filepath)
    
    @staticmethod
    def write_xml(drawable, filepath):
        return drawable.write_xml(filepath)


class WFD:

    file_extension = ".wfd.xml"

    @staticmethod
    def from_xml_file(filepath):
        global current_game
        tree = ET.parse(filepath)
        gameTag = tree.getroot().tag
        current_game = SollumzGame.RDR1
        return Drawable(gameTag).from_xml_file(filepath)
    
    @staticmethod
    def write_xml(drawable, filepath):
        return drawable.write_xml(filepath)


class Texture(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.name = TextProperty("Name", "")


class TextureDictionaryList(ListProperty):
    list_type = Texture
    tag_name = "TextureDictionary"


class RDRTextureDictionaryList(ElementTree, AbstractClass):
    tag_name = "TextureDictionary"

    def __init__(self) -> None:
        super().__init__()
        self.version = AttributeProperty("version", 1)
        self.textures = []
    
    @classmethod
    def from_xml(cls: Element, element: Element):
        new = super().from_xml(element)
        texs = element.find("Textures")
        if texs is not None:
            texs = texs.findall("Item")
            for tex in texs:
                texitem = Texture.from_xml(tex)
                if texitem:
                    texitem.tag_name = "Item"
                    new.textures.append(texitem)
        return new
    
    
    def to_xml(self):
        
        element = super().to_xml()
        texs = ET.Element("Textures")
        for value in self.textures:
            item = ET.Element("Item")
            name = ET.Element("Name")
            name.text = value.name
            flags = ET.Element("Flags")
            flags.set("value", str(value.flags))
            item.append(name)
            item.append(flags)
            texs.append(item)
        element.append(texs)
        return element
        

class ShaderParameter(ElementTree, AbstractClass):
    tag_name = "Item"

    @property
    @abstractmethod
    def type():
        raise NotImplementedError

    def __init__(self):
        super().__init__()
        self.name = AttributeProperty("name")
        self.type = AttributeProperty("type", self.type)


class TextureShaderParameter(ShaderParameter):
    type = "Texture"

    def __init__(self):
        super().__init__()

        if current_game == SollumzGame.RDR1:
            self.texture_name = AttributeProperty("texture", "")
            self.index = AttributeProperty("index", 0)
            self.flags = AttributeProperty("flags", 0)

    def __hash__(self) -> int:
        return hash((self.name, self.type, self.texture_name))


class VectorShaderParameter(ShaderParameter):
    type = "Vector"

    def __init__(self):
        super().__init__()
        self.x = AttributeProperty("x", 0)
        self.y = AttributeProperty("y", 0)
        self.z = AttributeProperty("z", 0)
        self.w = AttributeProperty("w", 0)

    def __hash__(self) -> int:
        return hash((self.name, self.type, self.x, self.y, self.z, self.w))


class ArrayShaderParameter(ShaderParameter):
    type = "Array"

    def __init__(self):
        super().__init__()
        self.values = []

    @staticmethod
    def from_xml(element: ET.Element):
        new = super(ArrayShaderParameter,
                    ArrayShaderParameter).from_xml(element)

        for item in element:
            new.values.append(Vector4Property.from_xml(item).value)

        return new

    def to_xml(self):
        element = super().to_xml()

        for value in self.values:
            child_elem = Vector4Property("Value", value).to_xml()
            element.append(child_elem)

        return element

    def __hash__(self) -> int:
        values_unpacked = [x for vector in self.values for x in [
            vector.x, vector.y, vector.z, vector.w]]
        return hash((self.name, self.type, *values_unpacked))


class CBufferShaderParameter(ShaderParameter):
    type = "CBuffer"

    def __init__(self):
        super().__init__()

    @staticmethod
    def from_xml(element: ET.Element):
        new = super(CBufferShaderParameter,
                    CBufferShaderParameter).from_xml(element)
        for item in element.attrib:
            val = element.attrib[item]
            if item not in ("name", "type", "value_type"):
                val = float(element.attrib[item])
            setattr(new, item, val)
        return new

    def to_xml(self):
        element = super().to_xml()
        element.set("buffer", str(int(self.buffer)))
        element.set("offset", str(int(self.offset)))
        element.set("length", str(self.length))

        if hasattr(self, "x") and self.x is not None:
            element.set("x", str(self.x))
        if hasattr(self, "y") and self.y is not None:
            element.set("y", str(self.y))
        if hasattr(self, "z") and self.z is not None:
            element.set("z", str(self.z))
        if hasattr(self, "w") and self.w is not None:
            element.set("w", str(self.w))
        
        if hasattr(self, "value") and self.value is not None:
            array_element = ET.SubElement(element, "Array")
            for value_tuple in self.value:
                item_element = ET.SubElement(array_element, "Item")
                item_element.set("x", str(value_tuple[0]))
                item_element.set("y", str(value_tuple[1]))
                item_element.set("z", str(value_tuple[2]))
                item_element.set("w", str(value_tuple[3]))

        return element

    def __hash__(self) -> int:
        return hash((self.name, self.type, self.length, self.x))


class SamplerShaderParameter(ShaderParameter):
    type = "Sampler"

    def __init__(self):
        super().__init__()
        self.index = AttributeProperty("index", 0)
        self.x = AttributeProperty("sampler", 0)

    def __hash__(self) -> int:
        return hash((self.name, self.type, self.index, self.x))


class UnknownShaderParameter(ShaderParameter):
    type = "Unknown"

    def __init__(self):
        super().__init__()
        self.index = AttributeProperty("index", 0)

    def __hash__(self) -> int:
        return hash((self.name, self.type, self.index))


class ParametersList(ListProperty):
    list_type = ShaderParameter
    tag_name = "Parameters"

    @staticmethod
    def from_xml(element: ET.Element):
        new = ParametersList()

        for child in element.iter():
            if "type" in child.attrib:
                param_type = child.get("type")
                if param_type == TextureShaderParameter.type:
                    new.value.append(TextureShaderParameter.from_xml(child))
                if param_type == VectorShaderParameter.type:
                    new.value.append(VectorShaderParameter.from_xml(child))
                if param_type == ArrayShaderParameter.type:
                    new.value.append(
                        ArrayShaderParameter.from_xml(child))

        return new

    def __hash__(self) -> int:
        return hash(tuple(hash(param) for param in self.value))


class RDRShaderParameter(ElementTree, AbstractClass):
    tag_name = "Item"

    @property
    @abstractmethod
    def type():
        raise NotImplementedError

    def __init__(self):
        super().__init__()
        self.name = AttributeProperty("name")
        self.type = AttributeProperty("type", self.type)


class RDRParametersList(ListProperty):
    list_type = ShaderParameter
    tag_name = "Items"

    @staticmethod
    def from_xml(element: ET.Element):
        new = RDRParametersList()

        for child in element.iter():
            if "type" in child.attrib:
                param_type = child.get("type")
                
                if param_type == TextureShaderParameter.type:
                    new.value.append(TextureShaderParameter.from_xml(child))
                if param_type == VectorShaderParameter.type:
                    new.value.append(VectorShaderParameter.from_xml(child))
                if param_type == ArrayShaderParameter.type:
                    new.value.append(ArrayShaderParameter.from_xml(child))
                if param_type == CBufferShaderParameter.type:
                    new.value.append(CBufferShaderParameter.from_xml(child))
                if param_type == SamplerShaderParameter.type:
                    new.value.append(SamplerShaderParameter.from_xml(child))
                if param_type == UnknownShaderParameter.type:
                    new.value.append(UnknownShaderParameter.from_xml(child))

        return new

    def __hash__(self) -> int:
        return hash(tuple(hash(param) for param in self.value))


class RDRParameters(ElementTree):
    tag_name = "Parameters"

    def __init__(self):
        self.buffer_size = TextProperty("BufferSizes", "")
        self.items = RDRParametersList()


class Shader(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.name = TextProperty("Name", "")
        if current_game == SollumzGame.RDR1:
            self.draw_bucket = ValueProperty("DrawBucket", 0)
            self.parameters = RDRParameters()

    def __hash__(self) -> int:
        params_elem = self.get_element("parameters")
        return hash((hash(self.name), hash(self.filename), hash(self.render_bucket), hash(params_elem)))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Shader):
            return False

        return hash(self) == hash(other)


class ShadersList(ListProperty):
    list_type = Shader
    tag_name = "Shaders"


class ShaderGroup(ElementTree):
    tag_name = "ShaderGroup"

    def __init__(self):
        super().__init__()
        if current_game == SollumzGame.RDR1:
            self.texture_dictionary = RDRTextureDictionaryList()
        self.shaders = ShadersList()


class BoneIDProperty(ElementProperty):
    value_types = (list)

    def __init__(self, tag_name: str = "BoneIDs", value=None):
        super().__init__(tag_name, value or [])

    @classmethod
    def from_xml(cls, element: ET.Element):
        new = cls()
        if element.text:
            if current_game == SollumzGame.RDR1:
                txt = element.text.split()
                new.value = [int(id) for id in txt if id.strip()]
            else:
                txt = element.text.split(", ")
                new.value = []
                for id in txt:
                    new.value.append(int(id))
        return new

    def to_xml(self):
        element = ET.Element(self.tag_name)

        if not self.value:
            return None

        if current_game == SollumzGame.RDR1:
            element.text = " ".join(str(id) for id in self.value)
        else:
            element.text = ", ".join([str(id) for id in self.value])
        return element


class Bone(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        # make enum in the future with all of the specific bone names?
        self.name = TextProperty("Name", "")
        self.tag = ValueProperty("BoneId", 0)
        self.flags = FlagsProperty("Flags")
        self.index = ValueProperty("Index", 0)
        # by default if a bone don't have parent or sibling there should be -1 instead of 0
        self.parent_index = ValueProperty("ParentIndex", -1)

        if current_game == SollumzGame.RDR1:
            self.mirror_index = ValueProperty("MirrorIndex", 0)
            self.default_rotation = VectorProperty("DefaultRotation")
            self.global_offset = VectorProperty("GlobalOffset")
            self.rotation_min = VectorProperty("RotationMin", Vector((-math.pi, -math.pi, -math.pi)))
            self.rotation_max = VectorProperty("RotationMax", Vector((math.pi, math.pi, math.pi)))
            self.joint_data = ValueProperty("JointData", 0)
            self.sibling_index = ValueProperty("SiblingIndex", -1)
            self.last_sibling_index = ValueProperty("ChildIndex", -1)
            self.translation = VectorProperty("DefaultTranslation")

        self.rotation = QuaternionProperty("DefaultRotationQuat")
        self.scale = VectorProperty("DefaultScale")
        


class BonesList(ListProperty):
    list_type = Bone
    tag_name = "Bones"


class Skeleton(ElementTree):
    tag_name = "Skeleton"

    def __init__(self):
        super().__init__()
        if current_game == SollumzGame.RDR1:
            self.flags = ValueProperty("Flags", 10)
            self.joint_data = ValueProperty("JointData", 13814012)
        self.bones = BonesList("Bones")


class BoneLimit(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.bone_id = ValueProperty("BoneId", 0)
        self.min = VectorProperty("Min")
        self.max = VectorProperty("Max")


class RotationLimit(BoneLimit):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.unk_a = ValueProperty("UnknownA", 0)


class RotationLimitsList(ListProperty):
    list_type = RotationLimit
    tag_name = "RotationLimits"


class TranslationLimitsList(ListProperty):
    list_type = BoneLimit
    tag_name = "TranslationLimits"


class Joints(ElementTree):
    tag_name = "Joints"

    def __init__(self):
        super().__init__()
        self.rotation_limits = RotationLimitsList("RotationLimits")
        self.translation_limits = TranslationLimitsList("TranslationLimits")


class VertexLayoutList(ElementProperty):
    value_types = (list)
    tag_name = "Layout"

    def __init__(self, type: str = "GTAV1", value: list[str] = None):
        super().__init__(self.tag_name, value or [])
        self.type = type

    @classmethod
    def from_xml(cls, element: ET.Element):
        new = cls()
        new.type = element.get("type")
        for child in element:
            new.value.append(child.tag)
        return new

    def to_xml(self):
        element = ET.Element(self.tag_name)
        for item in self.value:
            element.append(ET.Element(item))
        return element


class VertexBuffer(ElementTree):
    # Dtypes for vertex buffer structured numpy array
    # Based off of CodeWalker.GameFiles.VertexTypeGTAV1
    VERT_ATTR_DTYPES = {
        "Position": ("Position", np.float32, 3),
        "BlendWeights": ("BlendWeights", np.uint32, 4),
        "BlendIndices": ("BlendIndices", np.uint32, 4),
        "Normal": ("Normal", np.float32, 3),
        "Colour0": ("Colour0", np.uint32, 4),
        "Colour1": ("Colour1", np.uint32, 4),
        "TexCoord0": ("TexCoord0", np.float32, 2),
        "TexCoord1": ("TexCoord1", np.float32, 2),
        "TexCoord2": ("TexCoord2", np.float32, 2),
        "TexCoord3": ("TexCoord3", np.float32, 2),
        "TexCoord4": ("TexCoord4", np.float32, 2),
        "TexCoord5": ("TexCoord5", np.float32, 2),
        "TexCoord6": ("TexCoord6", np.float32, 2),
        "TexCoord7": ("TexCoord7", np.float32, 2),
        "Tangent": ("Tangent", np.float32, 4),
    }

    tag_name = "VertexBuffer"

    def __init__(self):
        super().__init__()
        self.flags = ValueProperty("Flags", 0)
        self.data: Optional[NDArray] = None

        self.layout = VertexLayoutList()

    @classmethod
    def from_xml(cls, element: ET.Element):
        new = super().from_xml(element)

        data_elem = element.find("Data")
        data2_elem = element.find("Data2")

        if data_elem is None and data2_elem is not None:
            data_elem = data2_elem

        if data_elem is None or not data_elem.text:
            return new

        new._load_data_from_str(data_elem.text)

        return new

    def to_xml(self):
        self.layout = self.data.dtype.names
        element = super().to_xml()

        if self.data is None:
            return element

        data_elem = ET.Element("Data")
        data_elem.text = self._data_to_str()

        element.append(data_elem)

        return element

    def _load_data_from_str(self, _str: str):
        struct_dtype = np.dtype([self.VERT_ATTR_DTYPES[attr_name]
                                 for attr_name in self.layout])

        self.data = np.loadtxt(io.StringIO(_str), dtype=struct_dtype)

    def _data_to_str(self):
        vert_arr = self.data

        FLOAT_FMT = "%.7f"
        INT_FMT = "%.0u"
        ATTR_SEP = "   "

        formats: list[str] = []

        for field_name in vert_arr.dtype.names:
            attr_dtype = vert_arr.dtype[field_name].base
            column = vert_arr[field_name]

            attr_fmt = INT_FMT if attr_dtype == np.uint32 else FLOAT_FMT
            formats.append(" ".join([attr_fmt] * column.shape[1]))

        fmt = ATTR_SEP.join(formats)
        vert_arr_2d = np.column_stack(
            [vert_arr[name] for name in vert_arr.dtype.names])

        return np_arr_to_str(vert_arr_2d, fmt)


class IndexBuffer(ElementTree):
    tag_name = "IndexBuffer"

    def __init__(self):
        super().__init__()
        self.data: Optional[NDArray] = None

    @classmethod
    def from_xml(cls, element: ET.Element):
        new = cls()

        data_elem = element.find("Data")

        if data_elem is None or not data_elem.text:
            return new

        new.data = np.fromstring(data_elem.text, sep=" ", dtype=np.uint32)
        return new

    def to_xml(self):
        element = ET.Element(self.tag_name)

        if self.data is None:
            return element

        data_elem = ET.Element("Data")
        data_elem.text = self._inds_to_str()

        element.append(data_elem)

        return element

    def _inds_to_str(self):
        indices_arr = self.data

        num_inds = len(indices_arr)

        # Get number of rows that can be split into 24 columns
        num_divisble_inds = num_inds - (num_inds % 24)
        num_rows = int(num_divisble_inds / 24)

        indices_arr_2d = indices_arr[:num_divisble_inds].reshape(
            (num_rows, 24))

        index_buffer_str = np_arr_to_str(indices_arr_2d, fmt="%.0u")
        # Add the last row
        last_row_str = np_arr_to_str(
            indices_arr[num_divisble_inds:], fmt="%.0u")

        return f"{index_buffer_str}\n{last_row_str}"


class Geometry(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.bounding_box_min = VectorProperty("BoundingBoxMin")
        self.bounding_box_max = VectorProperty("BoundingBoxMax")

        if current_game == SollumzGame.RDR1:
            self.shader_index = ValueProperty("ShaderID", 0)
            self.bone_ids = BoneIDProperty()
            self.vertex_layout = VertexLayout()
            self.vertices = VerticesProperty("Vertices")
            self.indices = IndicesProperty("Indices")


class GeometriesList(ListProperty):
    list_type = Geometry
    tag_name = "Geometries"


class DrawableModel(ElementTree):
    tag_name = "Item"

    def __init__(self):
        super().__init__()
        self.flags = ValueProperty("Flags", 0)
        self.has_skin = ValueProperty("HasSkin", 0)
        self.bone_index = ValueProperty("BoneIndex", 0)
        
        if current_game == SollumzGame.RDR1:
            self.matrix_count = ValueProperty("BoneCount", 0)
            self.bounding_box_min = VectorProperty("BoundingBoxMin")
            self.bounding_box_max = VectorProperty("BoundingBoxMax")

        self.geometries = GeometriesList()    


class DrawableModelList(ListProperty):
    list_type = DrawableModel
    tag_name = "DrawableModels"


class LodModelsList(ListProperty):
    list_type = DrawableModel
    tag_name = "Models"


class LodList(ElementTree):
    tag_name = "LodHigh"

    def __init__(self, tag_name: str = "LodHigh"):
        self.tag_name = tag_name
        super().__init__()
        self.models = LodModelsList()


class Drawable(ElementTree, AbstractClass):
    tag_name = "Drawable"

    @property
    def is_empty(self) -> bool:
        return len(self.all_models) == 0

    @property
    def all_geoms(self) -> list[Geometry]:
        return [geom for model in self.all_models for geom in model.geometries]

    @property
    def all_models(self) -> list[DrawableModel]:
        return self.drawable_models_high + self.drawable_models_med + self.drawable_models_low + self.drawable_models_vlow

    def __init__(self, tag_name: str = "Drawable"):
        self.tag_name = tag_name
        super().__init__()
        # Only in fragment drawables
        self.game = current_game
        self.matrix = MatrixProperty("Matrix")
        self.matrices = DrawableMatrices("Matrices")

        self.name = TextProperty("Name", "")
        self.hash = TextProperty("Hash", "")
        self.bounding_sphere_center = VectorProperty("BoundingSphereCenter")
        self.bounding_sphere_radius = ValueProperty("BoundingSphereRadius")
        self.bounding_box_min = VectorProperty("BoundingBoxMin")
        self.bounding_box_max = VectorProperty("BoundingBoxMax")
        self.lod_dist_high = ValueProperty("LodDistHigh", 0)  # 9998?
        self.lod_dist_med = ValueProperty("LodDistMed", 0)  # 9998?
        self.lod_dist_low = ValueProperty("LodDistLow", 0)  # 9998?
        self.lod_dist_vlow = ValueProperty("LodDistVlow", 0)  # 9998?
        self.flags_high = ValueProperty("FlagsHigh", 0)
        self.flags_med = ValueProperty("FlagsMed", 0)
        self.flags_low = ValueProperty("FlagsLow", 0)
        self.flags_vlow = ValueProperty("FlagsVlow", 0)
        self.shader_group = ShaderGroup()
        self.skeleton = Skeleton()

        if current_game == SollumzGame.RDR1:
            self.version = AttributeProperty("version", 0)
            self.drawable_models_high = LodList("LodHigh")
            self.drawable_models_med = LodList("LodMed")
            self.drawable_models_low = LodList("LodLow")
            self.drawable_models_vlow = LodList("LodVeryLow")
        
        self.bounds = []
        
        # For merging hi Drawables after import
        self.hi_models: list[DrawableModel] = []

    @classmethod
    def from_xml(cls, element: ET.Element):
        new = super().from_xml(element)
        return new

    def to_xml(self):
        element = super().to_xml()
        return element


class RDR1VisualDictionary(ElementTree, AbstractClass):
    tag_name = "RDR1VisualDictionary"

    def __init__(self) -> None:
        super().__init__()
        self.game = current_game
        self.version = AttributeProperty("version", 0)
        self.drawables = []

    @classmethod
    def from_xml(cls, element: ET.Element):
        new = super().from_xml(element)
        drawables = element.findall("Drawables")

        for item in drawables:
            drawable_items = item.findall("Item")
            for child in drawable_items:
                drawable = Drawable.from_xml(child)
                drawable.tag_name = "Drawable"
                new.drawables.append(drawable)
        return new
    
    def to_xml(self):   
        element = ET.Element(self.tag_name)
        element.set("version", str(self.version))
        drawables_element = ET.SubElement(element, "Drawables")
        for drawable in self.drawables:
            if isinstance(drawable, Drawable):
                drawable_element = drawable.to_xml()
                item_element = ET.SubElement(drawables_element, "Item")
                for child in drawable_element:
                    item_element.append(child)
            else:
                raise TypeError(
                    f"{type(self).__name__}s can only hold '{Drawable.__name__}' objects, not '{type(drawable)}'!")

        return element


class DrawableMatrices(ElementProperty):
    value_types = (list)

    def __init__(self, tag_name: str = "Matrices", value: list[Matrix] = None):
        super().__init__(tag_name, value)
        self.value = value or []

    @classmethod
    def from_xml(cls, element: Element):
        # Import not needed (this should be eventually calculated in CW anyway)
        return cls()

    def to_xml(self):
        if self.value is None or len(self.value) == 0:
            return

        elem = ET.Element("Matrices", attrib={"capacity": "64"})

        for mat in self.value:
            mat_prop = MatrixProperty("Item", mat)
            mat_elem = mat_prop.to_xml()
            mat_elem.attrib["id"] = "0"

            elem.append(mat_elem)

        return elem


class BonePropertiesManager:
    dictionary_xml = os.path.join(
        os.path.dirname(__file__), "BoneProperties.xml")
    bones = {}

    @staticmethod
    def load_bones():
        tree = ET.parse(BonePropertiesManager.dictionary_xml)
        for node in tree.getroot():
            bone = Bone.from_xml(node)
            BonePropertiesManager.bones[bone.name] = bone


BonePropertiesManager.load_bones()
