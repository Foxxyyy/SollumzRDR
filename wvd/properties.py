import bpy
import os
from typing import Optional
from ..tools.blenderhelper import lod_level_enum_flag_prop_factory
from ..sollumz_helper import find_sollumz_parent
from ..sollumz_properties import SOLLUMZ_UI_NAMES, SollumzGame, items_from_enums, TextureUsage, TextureFormat, LODLevel, SollumType, LightType, FlagPropertyGroup, TimeFlags
from ..wfd.shader_materials import rdr1_shadermats
from .render_bucket import RenderBucket, RenderBucketEnumItems
from bpy.app.handlers import persistent
from bpy.path import basename


class ShaderOrderItem(bpy.types.PropertyGroup):
    # For drawable shader order list
    index: bpy.props.IntProperty(min=0)
    name: bpy.props.StringProperty()
    filename: bpy.props.StringProperty()


class DrawableShaderOrder(bpy.types.PropertyGroup):
    items: bpy.props.CollectionProperty(type=ShaderOrderItem)
    active_index: bpy.props.IntProperty(min=0)

    def get_active_shader_item_index(self) -> int:
        return self.items[self.active_index].index

    def change_shader_index(self, old: int, new: int):
        if new >= len(self.items):
            return

        list_ind = self.active_index

        for i, item in enumerate(self.items):
            if item.index == new:
                item.index = old
            elif item.index == old:
                item.index = new
                list_ind = i

        self.active_index = list_ind


class DrawableProperties(bpy.types.PropertyGroup):
    lod_dist_high: bpy.props.FloatProperty(
        min=0, max=10000, default=9999, name="Lod Distance High")
    lod_dist_med: bpy.props.FloatProperty(
        min=0, max=10000, default=9999, name="Lod Distance Med")
    lod_dist_low: bpy.props.FloatProperty(
        min=0, max=10000, default=9999, name="Lod Distance Low")
    lod_dist_vlow: bpy.props.FloatProperty(
        min=0, max=10000, default=9999, name="Lod Distance Vlow")

    shader_order: bpy.props.PointerProperty(type=DrawableShaderOrder)


class DrawableModelProperties(bpy.types.PropertyGroup):
    render_mask: bpy.props.IntProperty(name="Render Mask", default=205)
    flags: bpy.props.IntProperty(name="Flags", default=0)
    sollum_lod: bpy.props.EnumProperty(
        items=items_from_enums(
            [LODLevel.HIGH, LODLevel.MEDIUM, LODLevel.LOW, LODLevel.VERYLOW]),
        name="LOD Level",
        default="sollumz_high"
    )


class SkinnedDrawableModelProperties(bpy.types.PropertyGroup):
    very_high: bpy.props.PointerProperty(type=DrawableModelProperties)
    high: bpy.props.PointerProperty(type=DrawableModelProperties)
    medium: bpy.props.PointerProperty(type=DrawableModelProperties)
    low: bpy.props.PointerProperty(type=DrawableModelProperties)
    very_low: bpy.props.PointerProperty(type=DrawableModelProperties)

    def get_lod(self, lod_level: LODLevel) -> DrawableModelProperties:
        if lod_level == LODLevel.VERYHIGH:
            return self.very_high
        elif lod_level == LODLevel.HIGH:
            return self.high
        elif lod_level == LODLevel.MEDIUM:
            return self.medium
        elif lod_level == LODLevel.LOW:
            return self.low
        elif lod_level == LODLevel.VERYLOW:
            return self.very_low


class ShaderProperties(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty(min=0)

    renderbucket: bpy.props.EnumProperty(
        name="Render Bucket", items=RenderBucketEnumItems,
        default=RenderBucket.OPAQUE.name
    )
    filename: bpy.props.StringProperty(
        name="Shader Filename", default="default.sps")
    name: bpy.props.StringProperty(name="Shader Name", default="default")


class TextureProperties(bpy.types.PropertyGroup):
    embedded: bpy.props.BoolProperty(name="Embedded", default=False)
    usage: bpy.props.EnumProperty(
        items=items_from_enums(TextureUsage),
        name="Usage",
        default=TextureUsage.UNKNOWN
    )

    format: bpy.props.EnumProperty(
        items=items_from_enums(TextureFormat),
        name="Format",
        default=TextureFormat.DXT1
    )


class BoneFlag(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="")


class BoneProperties(bpy.types.PropertyGroup):
    @staticmethod
    def calc_tag_hash(bone_name: str) -> int:
        h = 0
        for char in str.upper(bone_name):
            char = ord(char)
            h = (h << 4) + char
            x = h & 0xF0000000

            if x != 0:
                h ^= x >> 24

            h &= ~x

        return h % 0xFE8F + 0x170

    def get_bone(self) -> Optional[bpy.types.Bone]:
        armature: bpy.types.Armature = self.id_data
        if armature is None or not isinstance(armature, bpy.types.Armature):
            return None

        # no direct way to access the Bone from a PropertyGroup so iterate the armature bones until we find ourselves
        for bone in armature.bones:
            if bone.bone_properties == self:
                return bone
        return None

    def calc_tag(self) -> Optional[int]:
        bone = self.get_bone()
        if bone is None:
            return None

        is_root = bone.parent is None
        tag = 0 if is_root else BoneProperties.calc_tag_hash(bone.name)
        return tag

    def get_tag(self) -> int:
        if self.use_manual_tag:
            return self.manual_tag

        tag = self.calc_tag()
        if tag is None:
            # fallback to manual tag if for some reason we are not in a bone
            tag = self.manual_tag
        return tag

    def set_tag(self, value: int):
        self.manual_tag = value
        self.use_manual_tag = value != self.calc_tag()

class ShaderMaterial(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty("Index")
    name: bpy.props.StringProperty("Name")

@persistent
def on_file_loaded(_):
    # Handler sets the default value of the ShaderMaterials collection on blend file load
    bpy.context.scene.shader_materials.clear()
    for index, mat in enumerate(rdr1_shadermats):
        item = bpy.context.scene.shader_materials.add()
        item.index = index
        item.name = mat.name


def get_light_type(self):
    if self.type == "POINT":
        return 1 if not self.is_capsule else 3
    elif self.type == "SPOT":
        return 2
    else:
        return 0


def set_light_type(self, value):
    if value == 1:
        self.type = "POINT"
        self.is_capsule = False
    elif value == 3:
        self.type = "POINT"
        self.is_capsule = True
    elif value == 2:
        self.type = "SPOT"
        self.is_capsule = False


def get_texture_name(self):
    if self.image:
        return os.path.splitext(basename(self.image.filepath))[0]
    return "None"


def get_model_properties(model_obj: bpy.types.Object, lod_level: LODLevel) -> DrawableModelProperties:
    drawable_obj = find_sollumz_parent(model_obj, SollumType.VISUAL_DICTIONARY)

    if drawable_obj is not None and model_obj.vertex_groups:
        return drawable_obj.skinned_model_properties.get_lod(lod_level)

    lod = model_obj.sollumz_lods.get_lod(lod_level)

    if lod is None or lod.mesh is None:
        raise ValueError(
            f"Failed to get Drawable Model properties: {model_obj.name} has no {SOLLUMZ_UI_NAMES[lod_level]} LOD!")

    return lod.mesh.drawable_model_properties

def updateShaderList(self, context):
    materials = rdr1_shadermats
    game = "sollumz_rdr2"  
    context.scene.shader_materials.clear()
        
    for index, mat in enumerate(materials):
        item = context.scene.shader_materials.add()
        item.index = index
        item.name = mat.name
        item.game = game

def register():
    bpy.types.Scene.sollum_shader_game_type = bpy.props.EnumProperty(
        items=items_from_enums(SollumzGame),
        name="(HIDDEN)Sollumz Game",
        description="Hidden property used to sync with global game selection",
        default=SollumzGame.RDR1,
        options={"HIDDEN"},
        update=updateShaderList
    )
    bpy.types.Scene.shader_material_index = bpy.props.IntProperty(
        name="Shader Material Index")  # MAKE ENUM WITH THE MATERIALS NAMES
    bpy.types.Scene.shader_materials = bpy.props.CollectionProperty(
        type=ShaderMaterial, name="Shader Materials")
    bpy.app.handlers.load_post.append(on_file_loaded)
    bpy.types.Object.drawable_properties = bpy.props.PointerProperty(
        type=DrawableProperties)
    bpy.types.Material.shader_properties = bpy.props.PointerProperty(
        type=ShaderProperties)
    bpy.types.ShaderNodeTexImage.texture_properties = bpy.props.PointerProperty(
        type=TextureProperties)
    bpy.types.ShaderNodeTexImage.sollumz_texture_name = bpy.props.StringProperty(
        name="Texture Name", description="Name of texture.", get=get_texture_name)

    # Store properties for the DrawableModel with HasSkin=1. This is so all skinned objects share
    # the same drawable model properties even when split by group. It seems there is only ever 1
    # DrawableModel with HasSkin=1 in any given Drawable.
    bpy.types.Object.skinned_model_properties = bpy.props.PointerProperty(
        type=SkinnedDrawableModelProperties)
    # DrawableModel properties stored per mesh for LOD system
    bpy.types.Mesh.drawable_model_properties = bpy.props.PointerProperty(
        type=DrawableModelProperties)
    # For backwards compatibility
    bpy.types.Object.drawable_model_properties = bpy.props.PointerProperty(
        type=DrawableModelProperties)

    bpy.types.Scene.create_seperate_drawables = bpy.props.BoolProperty(
        name="Separate Objects", description="Create a separate Drawable for each selected object")
    bpy.types.Scene.auto_create_embedded_col = bpy.props.BoolProperty(
        name="Auto-Embed Collision", description="Automatically create embedded static collision")
    bpy.types.Scene.center_drawable_to_selection = bpy.props.BoolProperty(
        name="Center to Selection", description="Center Drawable(s) to selection", default=True)

    bpy.types.Bone.bone_properties = bpy.props.PointerProperty(
        type=BoneProperties)
    bpy.types.Light.sollum_type = bpy.props.EnumProperty(
        items=items_from_enums(LightType),
        name="Light Type",
        default=LightType.POINT,
        options={"HIDDEN"},
        get=get_light_type,
        set=set_light_type
    )
    bpy.types.Light.is_capsule = bpy.props.BoolProperty()
    bpy.types.Scene.create_light_type = bpy.props.EnumProperty(
        items=[
            (LightType.POINT.value,
             SOLLUMZ_UI_NAMES[LightType.POINT], SOLLUMZ_UI_NAMES[LightType.POINT]),
            (LightType.SPOT.value,
             SOLLUMZ_UI_NAMES[LightType.SPOT], SOLLUMZ_UI_NAMES[LightType.SPOT]),
            (LightType.CAPSULE.value,
             SOLLUMZ_UI_NAMES[LightType.CAPSULE], SOLLUMZ_UI_NAMES[LightType.CAPSULE]),
        ],
        name="Light Type",
        default=LightType.POINT,
        options={"HIDDEN"}
    )
    bpy.types.Light.time_flags = bpy.props.PointerProperty(type=TimeFlags)
    bpy.types.Scene.sollumz_auto_lod_ref_mesh = bpy.props.PointerProperty(
        type=bpy.types.Mesh, name="Reference Mesh", description="The mesh to copy and decimate for each LOD level. You'd usually want to set this as the highest LOD then run the tool for all lower LODs")
    bpy.types.Scene.sollumz_auto_lod_levels = lod_level_enum_flag_prop_factory()
    bpy.types.Scene.sollumz_auto_lod_decimate_step = bpy.props.FloatProperty(
        name="Decimate Step", min=0.0, max=0.99, default=0.6)

    bpy.types.Scene.light_preset_index = bpy.props.IntProperty(name="Light Preset Index")
    bpy.types.Scene.sollumz_extract_lods_levels = lod_level_enum_flag_prop_factory()
    bpy.types.Scene.sollumz_extract_lods_parent_type = bpy.props.EnumProperty(name="Parent Type", items=(
        ("sollumz_extract_lods_parent_type_object", "Object", "Parent to an Object"),
        ("sollumz_extract_lods_parent_type_collection",
         "Collection", "Parent to a Collection")
    ), default=0)


def unregister():
    del bpy.types.ShaderNodeTexImage.sollumz_texture_name
    del bpy.types.Scene.shader_material_index
    del bpy.types.Scene.shader_materials
    del bpy.types.Object.drawable_properties
    del bpy.types.Mesh.drawable_model_properties
    del bpy.types.Object.skinned_model_properties
    del bpy.types.Material.shader_properties
    del bpy.types.ShaderNodeTexImage.texture_properties
    del bpy.types.Bone.bone_properties
    del bpy.types.Light.light_properties
    del bpy.types.Scene.create_light_type
    del bpy.types.Light.time_flags
    del bpy.types.Light.light_flags
    del bpy.types.Light.is_capsule
    del bpy.types.Scene.light_presets
    del bpy.types.Scene.light_preset_index
    del bpy.types.Scene.create_seperate_drawables
    del bpy.types.Scene.auto_create_embedded_col
    del bpy.types.Scene.center_drawable_to_selection
    del bpy.types.Scene.sollumz_auto_lod_ref_mesh
    del bpy.types.Scene.sollumz_auto_lod_levels
    del bpy.types.Scene.sollumz_auto_lod_decimate_step
    del bpy.types.Scene.sollumz_extract_lods_levels
    del bpy.types.Scene.sollumz_extract_lods_parent_type

    bpy.app.handlers.load_post.remove(on_file_loaded)
