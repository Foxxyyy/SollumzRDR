import bpy
import os
import ast
from typing import Any
from .sollumz_properties import SollumType
from configparser import ConfigParser
from typing import Optional

PREFS_FILE_NAME = "sollumz_prefs.ini"


def _save_preferences(self, context):
    addon_prefs = get_addon_preferences(context)
    prefs_path = get_prefs_path()

    config = ConfigParser()
    prefs_dict = _get_data_block_as_dict(addon_prefs)
    main_prefs: dict[str, Any] = {}

    for key, value in prefs_dict.items():
        if isinstance(value, bpy.types.PropertyGroup):
            config[key] = _get_data_block_as_dict(value)
            continue

        main_prefs[key] = value

    config["main"] = main_prefs

    with open(prefs_path, "w") as f:
        config.write(f)


class SollumzExportSettings(bpy.types.PropertyGroup):
    limit_to_selected: bpy.props.BoolProperty(
        name="Limit to Selected",
        description="Export selected and visible objects only",
        default=True,
        update=_save_preferences
    )

    auto_calculate_inertia: bpy.props.BoolProperty(
        name="Auto Calculate Inertia",
        description="Automatically calculate inertia for physics objects (applies to yfts and ydrs too)",
        default=False,
        update=_save_preferences
    )

    auto_calculate_volume: bpy.props.BoolProperty(
        name="Auto Calculate Volume",
        description="Automatically calculate volume for physics objects (applies to yfts and ydrs too)",
        default=False,
        update=_save_preferences
    )

    exclude_skeleton: bpy.props.BoolProperty(
        name="Exclude Skeleton",
        description="Exclude skeleton from export. Usually done with mp ped components",
        default=False,
        update=_save_preferences
    )

    export_with_ytyp: bpy.props.BoolProperty(
        name="Export with ytyp",
        description="Exports a .ytyp.xml with an archetype for every drawable or drawable dictionary being exported",
        default=False,
        update=_save_preferences
    )

    ymap_exclude_entities: bpy.props.BoolProperty(
        name="Exclude Entities",
        description="If enabled, ignore all Entities from the selected ymap(s)",
        default=False,
        update=_save_preferences
    )

    ymap_box_occluders: bpy.props.BoolProperty(
        name="Exclude Box Occluders",
        description="If enabled, ignore all Box occluders from the selected ymap(s)",
        default=False,
        update=_save_preferences
    )

    ymap_model_occluders: bpy.props.BoolProperty(
        name="Exclude Model Occluders",
        description="If enabled, ignore all Model occluders from the selected ymap(s)",
        default=False,
        update=_save_preferences
    )

    ymap_car_generators: bpy.props.BoolProperty(
        name="Exclude Car Generators",
        description="If enabled, ignore all Car Generators from the selected ymap(s)",
        default=False,
        update=_save_preferences
    )

    export_lods: bpy.props.EnumProperty(
        name="Toggle LODs",
        description="Toggle LODs to export",
        options={"ENUM_FLAG"},
        default=({"sollumz_export_very_high", "sollumz_export_main_lods"}),
        items=(
            ("sollumz_export_very_high", "Very High",
             "Export Very High LODs into a _hi.yft"),
            ("sollumz_export_main_lods", "High - Very Low",
             "Export all LODs except Very High")
        ),
        update=_save_preferences
    )

    apply_transforms: bpy.props.BoolProperty(
        name="Apply Parent Transforms",
        description="Apply Drawable/Fragment scale and rotation",
        default=False,
        update=_save_preferences
    )

    @property
    def export_hi(self):
        return "sollumz_export_very_high" in self.export_lods

    @property
    def export_non_hi(self):
        return "sollumz_export_main_lods" in self.export_lods


class SollumzImportSettings(bpy.types.PropertyGroup):
    import_as_asset: bpy.props.BoolProperty(
        name="Import as asset",
        description="Create an asset from the .ydr/.yft high LOD",
        default=False,
        update=_save_preferences
    )

    import_with_hi: bpy.props.BoolProperty(
        name="Import with _hi",
        description="Import the selected .yft.xml with the <name>_hi.yft.xml placed in the very high LOD (must be in the same directory)",
        default=True,
        update=_save_preferences
    )

    split_by_group: bpy.props.BoolProperty(
        name="Split Mesh by Group",
        description="Splits the mesh by vertex groups",
        default=True,
        update=_save_preferences
    )

    import_ext_skeleton: bpy.props.BoolProperty(
        name="Import External Skeleton",
        description="Imports the first found yft skeleton in the same folder as the selected file",
        default=False,
        update=_save_preferences
    )

    ymap_skip_missing_entities: bpy.props.BoolProperty(
        name="Skip Missing Entities",
        description="If enabled, missing entities wont be created as an empty object",
        default=True,
        update=_save_preferences
    )

    ymap_exclude_entities: bpy.props.BoolProperty(
        name="Exclude Entities",
        description="If enabled, ignore all entities from the selected ymap(s)",
        default=False,
        update=_save_preferences
    )

    ymap_box_occluders: bpy.props.BoolProperty(
        name="Exclude Box Occluders",
        description="If enabled, ignore all Box occluders from the selected ymap(s)",
        default=False,
        update=_save_preferences
    )

    ymap_model_occluders: bpy.props.BoolProperty(
        name="Exclude Model Occluders",
        description="If enabled, ignore all Model occluders from the selected ymap(s)",
        default=False,
        update=_save_preferences
    )

    ymap_car_generators: bpy.props.BoolProperty(
        name="Exclude Car Generators",
        description="If enabled, ignore all Car Generators from the selected ymap(s)",
        default=False,
        update=_save_preferences
    )

    ymap_instance_entities: bpy.props.BoolProperty(
        name="Instance Entities",
        description="If enabled, instance all entities from the selected ymap(s).",
        default=False,
    )


class SollumzAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__.split(".")[0]

    scale_light_intensity: bpy.props.BoolProperty(
        name="Scale Light Intensity",
        description="Scale light intensity by 500 on import/export",
        default=True,
        update=_save_preferences
    )

    show_vertex_painter: bpy.props.BoolProperty(
        name="Show Vertex Painter",
        description="Show the Vertex Painter panel in General Tools (Includes Terrain Painter)",
        default=True,
        update=_save_preferences
    )

    extra_color_swatches: bpy.props.BoolProperty(
        name="Extra Vertex Color Swatches",
        description="Add 3 extra color swatches to the Vertex Painter Panel (Max 6)",
        default=True,
        update=_save_preferences
    )

    sollumz_icon_header: bpy.props.BoolProperty(
        name="Show Sollumz icon",
        description="Show the Sollumz icon in properties section headers",
        default=True,
        update=_save_preferences
    )
    use_text_name_as_mat_name: bpy.props.BoolProperty(
        name="Use Texture Name as Material Name",
        description="Use the name of the texture as the material name",
        default=True,
        update=_save_preferences
    )

    export_settings: bpy.props.PointerProperty(
        type=SollumzExportSettings, name="Export Settings")
    import_settings: bpy.props.PointerProperty(
        type=SollumzImportSettings, name="Import Settings")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "scale_light_intensity")
        layout.prop(self, "show_vertex_painter")
        layout.prop(self, "extra_color_swatches")
        layout.prop(self, "sollumz_icon_header")
        layout.prop(self, "use_text_name_as_mat_name")

    def register():
        _load_preferences()


def get_addon_preferences(context: Optional[bpy.types.Context] = None) -> SollumzAddonPreferences:
    return context.preferences.addons[__package__.split(".")[0]].preferences


def get_import_settings(context: Optional[bpy.types.Context] = None) -> SollumzImportSettings:
    return get_addon_preferences(context or bpy.context).import_settings


def get_export_settings(context: Optional[bpy.types.Context] = None) -> SollumzExportSettings:
    return get_addon_preferences(context or bpy.context).export_settings


def _load_preferences():
    # Preferences are loaded via an ini file in <user_blender_path>/<version>/config/sollumz_prefs.ini
    addon_prefs = get_addon_preferences(bpy.context)

    if addon_prefs is None:
        return

    prefs_path = get_prefs_path()

    if not os.path.exists(prefs_path):
        return

    config = ConfigParser()
    config.read(prefs_path)

    for section in config.keys():
        if section == "DEFAULT":
            continue

        if section == "main":
            _apply_preferences(addon_prefs, config, section)
            continue

        if not hasattr(addon_prefs, section):
            print(
                f"Unknown preferences pointer property '{section}'! Skipping...")
            continue

        prop_group = getattr(addon_prefs, section)
        _apply_preferences(prop_group, config, section)


def _apply_preferences(data_block: bpy.types.ID, config: ConfigParser, section: str):
    for key in config[section].keys():
        if not hasattr(data_block, key):
            print(f"Unknown preference '{key}'! Skipping...")
            continue

        value_str = config.get(section, key)
        value = ast.literal_eval(value_str)

        setattr(data_block, key, value)


def _get_data_block_as_dict(data_block: bpy.types.ID):
    data_block_dict: dict[str, Any] = {}

    for key in data_block.__annotations__.keys():
        if not hasattr(data_block, key):
            continue

        value = getattr(data_block, key)
        data_block_dict[key] = value

    return data_block_dict


def get_prefs_path():
    return os.path.join(bpy.utils.user_resource(resource_type='CONFIG'), PREFS_FILE_NAME)


def register():
    bpy.utils.register_class(SollumzAddonPreferences)


def unregister():
    bpy.utils.unregister_class(SollumzAddonPreferences)
