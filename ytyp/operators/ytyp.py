import os
import traceback
import bpy
from bpy_extras.io_utils import ImportHelper
from ...sollumz_helper import SOLLUMZ_OT_base, has_embedded_textures, has_collision
from ...sollumz_properties import SOLLUMZ_UI_NAMES, ArchetypeType, AssetType, SollumType, SollumzGame, MapEntityType
from ...sollumz_operators import SelectTimeFlagsRange, ClearTimeFlags
from ...sollumz_preferences import get_export_settings
from ..utils import get_selected_ytyp, get_selected_archetype


class SOLLUMZ_OT_create_ytyp(SOLLUMZ_OT_base, bpy.types.Operator):
    """Add a ytyp to the project"""
    bl_idname = "sollumz.createytyp"
    bl_label = "Create YTYP"

    def run(self, context):
        item = context.scene.ytyps.add()
        index = len(context.scene.ytyps)
        item.name = f"YTYP.{index}"
        item.game = context.scene.sollum_game_type
        context.scene.ytyp_index = index - 1

        return True


class SOLLUMZ_OT_delete_ytyp(SOLLUMZ_OT_base, bpy.types.Operator):
    """Delete a ytyp from the project"""
    bl_idname = "sollumz.deleteytyp"
    bl_label = "Delete YTYP"

    @classmethod
    def poll(cls, context):
        return len(context.scene.ytyps) > 0

    def run(self, context):
        context.scene.ytyps.remove(context.scene.ytyp_index)
        context.scene.ytyp_index = max(context.scene.ytyp_index - 1, 0)
        # Force redraw of gizmos
        context.space_data.show_gizmo = context.space_data.show_gizmo

        return True


class SOLLUMZ_OT_create_archetype(SOLLUMZ_OT_base, bpy.types.Operator):
    """Add an archetype to the selected ytyp"""
    bl_idname = "sollumz.createarchetype"
    bl_label = "Create Archetype"

    @classmethod
    def poll(cls, context):
        return get_selected_ytyp(context) is not None

    def run(self, context):
        selected_ytyp = get_selected_ytyp(context)
        selected_ytyp.new_archetype()

        return True


class SOLLUMZ_OT_set_texturedictionary_for_all_archetypes(SOLLUMZ_OT_base, bpy.types.Operator):
    """Sets texture dictionary for all archetypes within the selected ytyp"""
    bl_idname = "sollumz.settexturedictionaryallarchs"
    bl_label = "Set to All Archetypes"

    @classmethod
    def poll(cls, context):
        return get_selected_ytyp(context) is not None

    def execute(self, context):
        selected_ytyp = get_selected_ytyp(context)
        for archetype in selected_ytyp.archetypes:
            if archetype.asset_type != AssetType.ASSETLESS:
                archetype.texture_dictionary = selected_ytyp.all_texture_dictionary

        return {'FINISHED'}


class SOLLUMZ_OT_set_loddist_for_all_archetypes(SOLLUMZ_OT_base, bpy.types.Operator):
    """Sets lod dist for all archetypes within the selected ytyp"""
    bl_idname = "sollumz.setloddistallarchs"
    bl_label = "Set to All Archetypes"

    @classmethod
    def poll(cls, context):
        return get_selected_ytyp(context) is not None

    def execute(self, context):
        selected_ytyp = get_selected_ytyp(context)
        for archetype in selected_ytyp.archetypes:
            if archetype.asset_type != AssetType.ASSETLESS:
                archetype.lod_dist = selected_ytyp.all_lod_dist

        return {'FINISHED'}


class SOLLUMZ_OT_set_entity_loddist_for_all_archetypes(bpy.types.Operator):
    """Sets entity lod dist for all entities in all within the selected MLO archetype"""
    bl_idname = "sollumz.setentityloddistallarchs"
    bl_label = "Set to All Entities"

    @classmethod
    def poll(cls, context):
        selected_archetype = get_selected_archetype(context)
        return selected_archetype is not None and selected_archetype.type == ArchetypeType.MLO and len(selected_archetype.entities) > 0

    def execute(self, context):
        selected_archetype = get_selected_archetype(context)
        for entity in selected_archetype.entities:
            entity.lod_dist = selected_archetype.all_entity_lod_dist

        return {'FINISHED'}


class SOLLUMZ_OT_set_hdtexturedist_for_all_archetypes(SOLLUMZ_OT_base, bpy.types.Operator):
    """Sets HD textures distance for all archetypes within the selected ytyp"""
    bl_idname = "sollumz.sethdtexturedistallarchs"
    bl_label = "Set to All Archetypes"

    @classmethod
    def poll(cls, context):
        return get_selected_ytyp(context) is not None

    def execute(self, context):
        selected_ytyp = get_selected_ytyp(context)
        for archetype in selected_ytyp.archetypes:
            if archetype.asset_type != AssetType.ASSETLESS:
                archetype.hd_texture_dist = selected_ytyp.all_hd_tex_dist

        return {'FINISHED'}


class SOLLUMZ_OT_set_flag_for_all_archetypes(SOLLUMZ_OT_base, bpy.types.Operator):
    """Sets flags for all archetypes within the selected ytyp"""
    bl_idname = "sollumz.setflagsallarchs"
    bl_label = "Set to All Archetypes"

    @classmethod
    def poll(cls, context):
        return get_selected_ytyp(context) is not None

    def execute(self, context):
        selected_ytyp = get_selected_ytyp(context)
        for archetype in selected_ytyp.archetypes:
            if archetype.asset_type != AssetType.ASSETLESS:
                archetype.flags.total = str(selected_ytyp.all_flags)

        return {'FINISHED'}


class SOLLUMZ_OT_create_archetype_from_selected(SOLLUMZ_OT_base, bpy.types.Operator):
    """Create archetype from selected"""
    bl_idname = "sollumz.createarchetypefromselected"
    bl_label = "Auto-Create From Selected"

    allowed_types = [SollumType.DRAWABLE,
                     SollumType.BOUND_COMPOSITE, SollumType.FRAGMENT, SollumType.DRAWABLE_DICTIONARY]

    @classmethod
    def poll(cls, context):
        return get_selected_ytyp(context) is not None

    def run(self, context):
        selected_objs = context.selected_objects
        found = False
        for obj in selected_objs:
            selected_ytyp = get_selected_ytyp(context)
            archetype_type = context.scene.create_archetype_type
            if not obj.sollum_type in self.allowed_types:
                continue
            if archetype_type == ArchetypeType.MLO:
                if selected_ytyp.game == SollumzGame.RDR:
                    item.unknown_1 = MapEntityType.INTERIOR_INSTANCE
                if obj.sollum_type != SollumType.BOUND_COMPOSITE:
                    self.message(
                        f"MLO asset '{obj.name}' must be a {SOLLUMZ_UI_NAMES[SollumType.BOUND_COMPOSITE]}!")
                    continue
            found = True
            item = selected_ytyp.new_archetype()

            item.name = obj.name
            item.asset = obj
            item.type = archetype_type
            item.texture_dictionary = obj.name if has_embedded_textures(
                obj) else ""
            drawable_dictionary = ""
            if obj.parent:
                if obj.parent.sollum_type == SollumType.DRAWABLE_DICTIONARY:
                    drawable_dictionary = obj.parent.name
            item.drawable_dictionary = drawable_dictionary
            item.physics_dictionary = obj.name if has_collision(obj) and obj.sollum_type != SollumType.FRAGMENT else ""

            if obj.sollum_type == SollumType.DRAWABLE:
                item.asset_type = AssetType.DRAWABLE
                if selected_ytyp.game == SollumzGame.RDR:
                    item.unknown_1 = MapEntityType.BUILDING
            elif obj.sollum_type == SollumType.DRAWABLE_DICTIONARY:
                item.asset_type = AssetType.DRAWABLE_DICTIONARY
            elif obj.sollum_type == SollumType.BOUND_COMPOSITE:
                item.asset_type = AssetType.ASSETLESS
            elif obj.sollum_type == SollumType.FRAGMENT:
                item.asset_type = AssetType.FRAGMENT
        if not found:
            self.message(
                f"No asset of type '{','.join([SOLLUMZ_UI_NAMES[type] for type in self.allowed_types])}' found!")
            return False
        return True


class SOLLUMZ_OT_delete_archetype(SOLLUMZ_OT_base, bpy.types.Operator):
    """Delete archetype from selected ytyp"""
    bl_idname = "sollumz.deletearchetype"
    bl_label = "Delete Archetype"

    @classmethod
    def poll(cls, context):
        selected_ytyp = get_selected_ytyp(context)
        return selected_ytyp is not None and len(selected_ytyp.archetypes) > 0

    def run(self, context):
        selected_ytyp = get_selected_ytyp(context)
        selected_ytyp.archetypes.remove(selected_ytyp.archetype_index)
        selected_ytyp.archetype_index = max(
            selected_ytyp.archetype_index - 1, 0)
        # Force redraw of gizmos
        context.space_data.show_gizmo = context.space_data.show_gizmo

        return True


class SOLLUMZ_OT_create_timecycle_modifier(SOLLUMZ_OT_base, bpy.types.Operator):
    """Add a timecycle modifier to the selected archetype"""
    bl_idname = "sollumz.createtimecyclemodifier"
    bl_label = "Create Timecycle Modifier"

    @classmethod
    def poll(cls, context):
        return get_selected_archetype(context) is not None

    def run(self, context):
        selected_archetype = get_selected_archetype(context)
        item = selected_archetype.new_tcm()
        item.name = f"Timecycle Modifier.{len(selected_archetype.timecycle_modifiers)}"
        return True


class SOLLUMZ_OT_delete_timecycle_modifier(SOLLUMZ_OT_base, bpy.types.Operator):
    """Delete timecycle modifier from selected archetype"""
    bl_idname = "sollumz.deletetimecyclemodifier"
    bl_label = "Delete Timecycle Modifier"

    @classmethod
    def poll(cls, context):
        selected_archetype = get_selected_archetype(context)
        return selected_archetype is not None and selected_archetype.timecycle_modifiers

    def run(self, context):
        selected_archetype = get_selected_archetype(context)
        selected_archetype.timecycle_modifiers.remove(
            selected_archetype.tcm_index)
        selected_archetype.tcm_index = max(selected_archetype.tcm_index - 1, 0)
        return True


class SOLLUMZ_OT_YTYP_TIME_FLAGS_select_range(SelectTimeFlagsRange, bpy.types.Operator):
    bl_idname = "sollumz.ytyp_time_flags_select_range"

    @classmethod
    def poll(cls, context):
        return get_selected_archetype(context) is not None

    def get_flags(self, context):
        return get_selected_archetype(context).time_flags


class SOLLUMZ_OT_YTYP_TIME_FLAGS_clear(ClearTimeFlags, bpy.types.Operator):
    bl_idname = "sollumz.ytyp_time_flags_clear"

    @classmethod
    def poll(cls, context):
        return get_selected_archetype(context) is not None

    def get_flags(self, context):
        return get_selected_archetype(context).time_flags