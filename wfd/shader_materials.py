from typing import Optional, NamedTuple
import bpy
from ..cwxml.shader import (
    ShaderManager,
    ShaderDef,
    ShaderParameterType,
    ShaderParameterSubtype,
    ShaderParameterFloatDef,
    ShaderParameterFloat2Def,
    ShaderParameterFloat3Def,
    ShaderParameterFloat4Def,
    ShaderParameterFloat4x4Def,
)
from ..sollumz_properties import MaterialType, SollumzGame
from ..tools.blenderhelper import find_bsdf_and_material_output
from ..tools.meshhelper import get_uv_map_name
from ..wfd.shader_materials_SHARED import ShaderBuilder, create_image_node, create_parameter_node, link_value_shader_parameters, link_normal, try_get_node, link_diffuse, create_decal_nodes
from ..shared.shader_nodes import SzShaderNodeParameter, SzShaderNodeParameterDisplayType
from .render_bucket import RenderBucket, bucket_mapping

class ShaderBuilder(NamedTuple):
    shader: ShaderDef
    filename: str
    material: bpy.types.Material
    node_tree: bpy.types.ShaderNodeTree
    bsdf: bpy.types.ShaderNodeBsdfPrincipled
    material_output: bpy.types.ShaderNodeOutputMaterial


class ShaderMaterial(NamedTuple):
    name: str
    ui_name: str
    value: str

rdr1_shadermats = []

for shader in ShaderManager._rdr1_shaders.values():
    name = shader.filename.replace(".sps", "").upper()

    rdr1_shadermats.append(ShaderMaterial(
        name, name.replace("_", " "), shader.filename))

def try_get_node(node_tree: bpy.types.NodeTree, name: str) -> Optional[bpy.types.Node]:
    """Gets a node by its name. Returns `None` if not found.
    Note, names are localized by Blender or can changed by the user, so
    this should only be used for names that Sollumz sets explicitly.
    """
    return node_tree.nodes.get(name, None)


def try_get_node_by_cls(node_tree: bpy.types.NodeTree, node_cls: type) -> Optional[bpy.types.Node]:
    """Gets a node by its type. Returns `None` if not found."""
    for node in node_tree.nodes:
        if isinstance(node, node_cls):
            return node

    return None


def get_child_nodes(node):
    child_nodes = []
    for input in node.inputs:
        for link in input.links:
            child = link.from_node
            if child in child_nodes:
                continue
            else:
                child_nodes.append(child)
    return child_nodes


def group_image_texture_nodes(node_tree):
    image_texture_nodes = [node for node in node_tree.nodes if node.type == "TEX_IMAGE"]

    if not image_texture_nodes:
        return

    image_texture_nodes.sort(key=lambda node: node.location.y)

    avg_x = min([node.location.x for node in image_texture_nodes])

    # adjust margin to change gap in between img nodes
    margin = 275
    current_y = min([node.location.y for node in image_texture_nodes]) - margin
    for node in image_texture_nodes:
        current_y += margin
        node.location.x = avg_x
        node.location.y = current_y

    # how far to the left the img nodes are
    group_offset = 400
    for node in image_texture_nodes:
        node.location.x -= group_offset
        node.location.y += group_offset


def get_loose_nodes(node_tree):
    loose_nodes = []
    for node in node_tree.nodes:
        no = False
        ni = False
        for output in node.outputs:
            for link in output.links:
                if link.to_node is not None and link.from_node is not None:
                    no = True
                    break
        for input in node.inputs:
            for link in input.links:
                if link.to_node is not None and link.from_node is not None:
                    ni = True
                    break
        if no == False and ni == False:
            loose_nodes.append(node)
    return loose_nodes


def organize_node_tree(b: ShaderBuilder):
    mo = b.material_output
    mo.location.x = 0
    mo.location.y = 0
    organize_node(mo)
    organize_loose_nodes(b.node_tree, 1000, 0)
    group_image_texture_nodes(b.node_tree)


def organize_node(node):
    child_nodes = get_child_nodes(node)
    if len(child_nodes) < 0:
        return

    level = node.location.y
    for child in child_nodes:
        child.location.x = node.location.x - 300
        child.location.y = level
        level -= 300
        organize_node(child)


def organize_loose_nodes(node_tree, start_x, start_y):
    loose_nodes = get_loose_nodes(node_tree)
    if len(loose_nodes) == 0:
        return

    grid_x = start_x
    grid_y = start_y

    for i, node in enumerate(loose_nodes):
        if i % 4 == 0:
            grid_x = start_x
            grid_y -= 150

        node.location.x = grid_x + node.width / 2
        node.location.y = grid_y - node.height / 2

        grid_x += node.width + 25


def get_tint_sampler_node(mat: bpy.types.Material) -> Optional[bpy.types.ShaderNodeTexImage]:
    nodes = mat.node_tree.nodes
    for node in nodes:
        if node.name == "TintPaletteSampler" and isinstance(node, bpy.types.ShaderNodeTexImage):
            return node

    return None


def get_detail_extra_sampler(mat):  # move to blenderhelper.py?
    nodes = mat.node_tree.nodes
    for node in nodes:
        if node.name == "Extra":
            return node
    return None


def create_tinted_shader_graph(obj: bpy.types.Object):
    tint_mats = get_tinted_mats(obj)

    if not tint_mats:
        return

    for mat in tint_mats:
        tint_sampler_node = get_tint_sampler_node(mat)
        palette_img = tint_sampler_node.image

        if tint_sampler_node is None:
            continue

        if mat.shader_properties.filename in ShaderManager.tint_colour1_shaders:
            input_color_attr_name = "Color 2"
        else:
            input_color_attr_name = "Color 1"

        tint_color_attr_name = f"TintColor ({palette_img.name})" if palette_img else "TintColor"
        tint_color_attr = obj.data.attributes.new(name=tint_color_attr_name, type="BYTE_COLOR", domain="CORNER")

        rename_tint_attr_node(mat.node_tree, name=tint_color_attr.name)

        create_tint_geom_modifier(obj, tint_color_attr.name, input_color_attr_name, palette_img)


def create_tint_geom_modifier(
    obj: bpy.types.Object,
    tint_color_attr_name: str,
    input_color_attr_name: Optional[str],
    palette_img: Optional[bpy.types.Image]
) -> bpy.types.NodesModifier:
    tnt_ng = create_tinted_geometry_graph()
    mod = obj.modifiers.new("GeometryNodes", "NODES")
    mod.node_group = tnt_ng

    # set input / output variables
    input_id = tnt_ng.interface.items_tree["Color Attribute"].identifier
    mod[input_id + "_attribute_name"] = input_color_attr_name if input_color_attr_name is not None else ""
    mod[input_id + "_use_attribute"] = True

    input_palette_id = tnt_ng.interface.items_tree["Palette Texture"].identifier
    mod[input_palette_id] = palette_img

    output_id = tnt_ng.interface.items_tree["Tint Color"].identifier
    mod[output_id + "_attribute_name"] = tint_color_attr_name
    mod[output_id + "_use_attribute"] = True

    return mod


def rename_tint_attr_node(node_tree: bpy.types.NodeTree, name: str):
    for node in node_tree.nodes:
        if not isinstance(node, bpy.types.ShaderNodeAttribute) or node.attribute_name != "TintColor":
            continue

        node.attribute_name = name
        return


def get_tinted_mats(obj: bpy.types.Object) -> list[bpy.types.Material]:
    if obj.data is None or not obj.data.materials:
        return []

    return [mat for mat in obj.data.materials if is_tint_material(mat)]


def obj_has_tint_mats(obj: bpy.types.Object) -> bool:
    if not obj.data.materials:
        return False

    mat = obj.data.materials[0]
    return is_tint_material(mat)


def is_tint_material(mat: bpy.types.Material) -> bool:
    return get_tint_sampler_node(mat) is not None


def link_geos(links, node1, node2):
    links.new(node1.inputs["Geometry"], node2.outputs["Geometry"])


def create_tinted_geometry_graph():  # move to blenderhelper.py?
    gnt = bpy.data.node_groups.new(name="TintGeometry", type="GeometryNodeTree")
    input = gnt.nodes.new("NodeGroupInput")
    output = gnt.nodes.new("NodeGroupOutput")

    # Create the necessary sockets for the node group
    gnt.interface.new_socket("Geometry", socket_type="NodeSocketGeometry", in_out="INPUT")
    gnt.interface.new_socket("Geometry", socket_type="NodeSocketGeometry", in_out="OUTPUT")
    gnt.interface.new_socket("Color Attribute", socket_type="NodeSocketVector", in_out="INPUT")
    in_palette = gnt.interface.new_socket("Palette (Preview)",
                                          description="Index of the tint palette to preview. Has no effect on export",
                                          socket_type="NodeSocketInt", in_out="INPUT")
    in_palette.min_value = 0
    gnt.interface.new_socket("Palette Texture", description="Should be the same as 'TintPaletteSampler' of the material",
                             socket_type="NodeSocketImage", in_out="INPUT")
    gnt.interface.new_socket("Tint Color", socket_type="NodeSocketColor", in_out="OUTPUT")

    # link input / output node to create geometry socket
    cptn = gnt.nodes.new("GeometryNodeCaptureAttribute")
    cptn.domain = "CORNER"
    cptn.data_type = "FLOAT_COLOR"
    gnt.links.new(input.outputs[0], cptn.inputs[0])
    gnt.links.new(cptn.outputs[0], output.inputs[0])

    # create and link texture node
    txtn = gnt.nodes.new("GeometryNodeImageTexture")
    txtn.interpolation = "Closest"
    gnt.links.new(input.outputs[3], txtn.inputs[0])
    gnt.links.new(cptn.outputs[3], txtn.inputs[1])
    gnt.links.new(txtn.outputs[0], output.inputs[1])

    # separate colour0
    sepn = gnt.nodes.new("ShaderNodeSeparateXYZ")
    gnt.links.new(input.outputs[1], sepn.inputs[0])

    # create math nodes
    mathns = []
    for i in range(9):
        mathns.append(gnt.nodes.new("ShaderNodeMath"))

    # Convert color attribute from linear to sRGB
    # Sollumz imports it as sRGB but accessing in the node tree gives you linear color
    # c1
    mathns[0].operation = "LESS_THAN"
    gnt.links.new(sepn.outputs[2], mathns[0].inputs[0])
    mathns[0].inputs[1].default_value = 0.003
    mathns[1].operation = "SUBTRACT"
    gnt.links.new(mathns[0].outputs[0], mathns[1].inputs[1])
    mathns[1].inputs[0].default_value = 1.0

    # r1
    mathns[2].operation = "MULTIPLY"
    gnt.links.new(sepn.outputs[2], mathns[2].inputs[0])
    mathns[2].inputs[1].default_value = 12.920
    mathns[3].operation = "MULTIPLY"
    gnt.links.new(mathns[2].outputs[0], mathns[3].inputs[0])
    gnt.links.new(mathns[0].outputs[0], mathns[3].inputs[1])

    # r2
    mathns[4].operation = "POWER"
    gnt.links.new(sepn.outputs[2], mathns[4].inputs[0])
    mathns[4].inputs[1].default_value = 0.417
    mathns[5].operation = "MULTIPLY"
    gnt.links.new(mathns[4].outputs[0], mathns[5].inputs[0])
    mathns[5].inputs[1].default_value = 1.055
    mathns[6].operation = "SUBTRACT"
    gnt.links.new(mathns[5].outputs[0], mathns[6].inputs[0])
    mathns[6].inputs[1].default_value = 0.055
    mathns[7].operation = "MULTIPLY"
    gnt.links.new(mathns[6].outputs[0], mathns[7].inputs[0])
    gnt.links.new(mathns[1].outputs[0], mathns[7].inputs[1])

    # add r1 and r2
    mathns[8].operation = "ADD"
    gnt.links.new(mathns[3].outputs[0], mathns[8].inputs[0])
    gnt.links.new(mathns[7].outputs[0], mathns[8].inputs[1])

    # Select palette row
    # uv.y = (palette_preview_index + 0.5) / img.height
    # uv.y = ((uv.y - 1) * -1)   ; flip_uv
    pal_add = gnt.nodes.new("ShaderNodeMath")
    pal_add.operation = "ADD"
    pal_add.inputs[1].default_value = 0.5
    pal_img_info = gnt.nodes.new("GeometryNodeImageInfo")
    pal_div = gnt.nodes.new("ShaderNodeMath")
    pal_div.operation = "DIVIDE"
    pal_flip_uv_sub = gnt.nodes.new("ShaderNodeMath")
    pal_flip_uv_sub.operation = "SUBTRACT"
    pal_flip_uv_sub.inputs[1].default_value = 1.0
    pal_flip_uv_mult = gnt.nodes.new("ShaderNodeMath")
    pal_flip_uv_mult.operation = "MULTIPLY"
    pal_flip_uv_mult.inputs[1].default_value = -1.0

    gnt.links.new(input.outputs[3], pal_img_info.inputs[0])
    gnt.links.new(input.outputs[2], pal_add.inputs[1])
    gnt.links.new(pal_add.outputs[0], pal_div.inputs[0])
    gnt.links.new(pal_img_info.outputs[1], pal_div.inputs[1])
    gnt.links.new(pal_div.outputs[0], pal_flip_uv_sub.inputs[0])
    gnt.links.new(pal_flip_uv_sub.outputs[0], pal_flip_uv_mult.inputs[0])

    # create and link vector
    comb = gnt.nodes.new("ShaderNodeCombineRGB")
    gnt.links.new(mathns[8].outputs[0], comb.inputs[0])
    gnt.links.new(pal_flip_uv_mult.outputs[0], comb.inputs[1])
    gnt.links.new(comb.outputs[0], cptn.inputs[3])

    return gnt
    node_tree = b.node_tree
    bsdf = b.bsdf
    dtltex2 = node_tree.nodes.new("ShaderNodeTexImage")
    dtltex2.name = "Extra"
    dtltex2.label = dtltex2.name
    ds = node_tree.nodes["detailmapscale"]
    dsu = node_tree.nodes["detailmapscaleu"]
    links = node_tree.links
    uv_map0 = node_tree.nodes[get_uv_map_name(0)]
    comxyz = node_tree.nodes.new("ShaderNodeCombineXYZ")
    mathns = []
    for _ in range(9):
        math = node_tree.nodes.new("ShaderNodeVectorMath")
        mathns.append(math)
    nrm = node_tree.nodes.new("ShaderNodeNormalMap")

    links.new(uv_map0.outputs[0], mathns[0].inputs[0])

    links.new(dsu.outputs["X"], comxyz.inputs[0])
    links.new(dsu.outputs["W"], comxyz.inputs[1])

    mathns[0].operation = "MULTIPLY"
    links.new(comxyz.outputs[0], mathns[0].inputs[1])
    links.new(mathns[0].outputs[0], dtltex2.inputs[0])

    mathns[1].operation = "MULTIPLY"
    mathns[1].inputs[1].default_value[0] = 3.17
    mathns[1].inputs[1].default_value[1] = 3.17
    links.new(mathns[0].outputs[0], mathns[1].inputs[0])
    links.new(mathns[1].outputs[0], dtltex.inputs[0])

    mathns[2].operation = "SUBTRACT"
    mathns[2].inputs[1].default_value[0] = 0.5
    mathns[2].inputs[1].default_value[1] = 0.5
    links.new(dtltex.outputs[0], mathns[2].inputs[0])

    mathns[3].operation = "SUBTRACT"
    mathns[3].inputs[1].default_value[0] = 0.5
    mathns[3].inputs[1].default_value[1] = 0.5
    links.new(dtltex2.outputs[0], mathns[3].inputs[0])

    mathns[4].operation = "ADD"
    links.new(mathns[2].outputs[0], mathns[4].inputs[0])
    links.new(mathns[3].outputs[0], mathns[4].inputs[1])

    mathns[5].operation = "MULTIPLY"
    links.new(mathns[4].outputs[0], mathns[5].inputs[0])
    links.new(ds.outputs["Y"], mathns[5].inputs[1])

    mathns[6].operation = "MULTIPLY"
    if spectex:
        links.new(spectex.outputs[1], mathns[6].inputs[0])
    links.new(mathns[5].outputs[0], mathns[6].inputs[1])

    mathns[7].operation = "MULTIPLY"
    mathns[7].inputs[1].default_value[0] = 1
    mathns[7].inputs[1].default_value[1] = 1
    links.new(mathns[6].outputs[0], mathns[7].inputs[0])

    mathns[8].operation = "ADD"
    links.new(mathns[7].outputs[0], mathns[8].inputs[0])
    links.new(bumptex.outputs[0], mathns[8].inputs[1])

    links.new(mathns[8].outputs[0], nrm.inputs[1])
    links.new(nrm.outputs[0], bsdf.inputs["Normal"])

def create_image_node(node_tree, param) -> bpy.types.ShaderNodeTexImage:
    imgnode = node_tree.nodes.new("ShaderNodeTexImage")
    imgnode.name = param.name
    imgnode.label = param.name
    imgnode.is_sollumz = True
    return imgnode

def link_diffuse(b: ShaderBuilder, imgnode):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links
    links.new(imgnode.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(imgnode.outputs["Alpha"], bsdf.inputs["Alpha"])

def link_diffuses(b: ShaderBuilder, tex1, tex2):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links
    rgb = node_tree.nodes.new("ShaderNodeMixRGB")
    links.new(tex1.outputs["Color"], rgb.inputs["Color1"])
    links.new(tex2.outputs["Color"], rgb.inputs["Color2"])
    links.new(tex2.outputs["Alpha"], rgb.inputs["Fac"])
    links.new(rgb.outputs["Color"], bsdf.inputs["Base Color"])
    return rgb

def link_normal(b: ShaderBuilder, nrmtex):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links

    separate_rgb_node = node_tree.nodes.new(type="ShaderNodeSeparateRGB")
    combine_rgb_node = node_tree.nodes.new(type="ShaderNodeCombineRGB")
    rgb_curves: bpy.types.ShaderNodeRGBCurve = node_tree.nodes.new("ShaderNodeRGBCurve")
    green_curves = rgb_curves.mapping.curves[1]
    green_curves.points[0].location = (0, 1)
    green_curves.points[1].location = (1, 0)

    links.new(nrmtex.outputs[0], separate_rgb_node.inputs[0]) # Connect normal map to Separate RGB
    links.new(separate_rgb_node.outputs[1], rgb_curves.inputs[1]) # Connect green channel to RGB Curves

    links.new(separate_rgb_node.outputs[0], combine_rgb_node.inputs[0])  # Red channel
    links.new(rgb_curves.outputs[0], combine_rgb_node.inputs[1]) # Inverted green channel
    links.new(nrmtex.outputs[1], combine_rgb_node.inputs[2]) # Alpha channel to blue
    links.new(combine_rgb_node.outputs[0], bsdf.inputs["Normal"]) # Connect Combine Normal to BSDF

def link_specular(b: ShaderBuilder, spctex):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links
    links.new(spctex.outputs["Color"], bsdf.inputs["Specular IOR Level"])

def link_value_shader_parameters(b: ShaderBuilder):
    shader = b.shader
    node_tree = b.node_tree
    links = node_tree.links

    bsdf = b.bsdf
    bmp = None
    spec_im = None
    spec_fm = None
    em_m = None

    for param in shader.parameters:
        if param.name == "bumpiness":
            bmp = node_tree.nodes["bumpiness"]
        elif param.name == "specularIntensityMult":
            spec_im = node_tree.nodes["specularIntensityMult"]
        elif param.name == "specularFalloffMult":
            spec_fm = node_tree.nodes["specularFalloffMult"]
        elif param.name == "emissiveMultiplier":
            em_m = node_tree.nodes["emissiveMultiplier"]

    if bmp:
        nm = try_get_node_by_cls(node_tree, bpy.types.ShaderNodeNormalMap)
        if nm:
            links.new(bmp.outputs["X"], nm.inputs[0])
    if spec_im:
        spec = try_get_node(node_tree, "SpecSampler")
        if spec:
            map = node_tree.nodes.new("ShaderNodeMapRange")
            map.inputs[2].default_value = 1
            map.inputs[4].default_value = 1
            map.clamp = True
            mult = node_tree.nodes.new("ShaderNodeMath")
            mult.operation = "MULTIPLY"
            links.new(spec.outputs[0], mult.inputs[0])
            links.new(map.outputs[0], mult.inputs[1])
            links.new(spec_im.outputs["X"], map.inputs[0])
            links.new(mult.outputs[0], bsdf.inputs["Specular IOR Level"])
    if spec_fm:
        map = node_tree.nodes.new("ShaderNodeMapRange")
        map.inputs[2].default_value = 512
        map.inputs[3].default_value = 1
        map.inputs[4].default_value = 0
        map.clamp = True
        links.new(spec_fm.outputs["X"], map.inputs[0])
        links.new(map.outputs[0], bsdf.inputs["Roughness"])
    if em_m:
        em = try_get_node_by_cls(node_tree, bpy.types.ShaderNodeEmission)
        if em:
            links.new(em_m.outputs["X"], em.inputs[1])

def create_uv_map_nodes(b: ShaderBuilder):
    """Creates a ``ShaderNodeUVMap`` node for each UV map used in the shader."""
    shader = b.shader
    node_tree = b.node_tree

    used_uv_maps = set(shader.uv_maps.values())
    for uv_map_index in used_uv_maps:
        uv_map = get_uv_map_name(uv_map_index)
        node = node_tree.nodes.new("ShaderNodeUVMap")
        node.name = uv_map
        node.label = uv_map
        node.uv_map = uv_map

def link_uv_map_nodes_to_textures(b: ShaderBuilder):
    """For each texture node, links the corresponding UV map to its input UV if it hasn't been linked already."""
    shader = b.shader
    node_tree = b.node_tree

    for tex_name, uv_map_index in shader.uv_maps.items():
        tex_node = node_tree.nodes[tex_name]
        uv_map_node = node_tree.nodes[get_uv_map_name(uv_map_index)]

        if tex_node.inputs[0].is_linked:
            # texture already linked when creating the node tree, skip it
            continue

        node_tree.links.new(uv_map_node.outputs[0], tex_node.inputs[0])

def create_tint_nodes(
    b: ShaderBuilder,
    diffuse_tex: bpy.types.ShaderNodeTexImage
):
    # create shader attribute node
    # TintColor attribute is filled by tint geometry nodes
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links
    attr = node_tree.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "TintColor"
    mix = node_tree.nodes.new("ShaderNodeMixRGB")
    # hacky shit here for now
    is_fully_black = all(value == 0.0 for value in attr.outputs[0].default_value)
    if is_fully_black:
        mix.inputs["Fac"].default_value = 0.0
    else:
        mix.inputs["Fac"].default_value = 0.95
    mix.blend_type = "MULTIPLY"
    links.new(attr.outputs["Color"], mix.inputs[2])
    links.new(diffuse_tex.outputs[0], mix.inputs[1])
    links.new(mix.outputs[0], bsdf.inputs["Base Color"])

def RDR_create_basic_shader_nodes(b: ShaderBuilder, game = SollumzGame.RDR1):
    shader = b.shader
    filename = b.filename
    mat = b.material
    node_tree = b.node_tree
    bsdf = b.bsdf

    texture = None
    texture2 = None
    texture3 = None
    bumptex = None

    for param in shader.parameters:
        match param.type:
            case ShaderParameterType.TEXTURE:
                imgnode = create_image_node(node_tree, param)
                if param.name == "texturesampler":
                    texture = imgnode
                elif param.name == "bumpsampler":
                    bumptex = imgnode
                elif param.name == "texturesampler2":
                    texture2 = imgnode
                elif param.name == "texturesampler3":
                    texture3 = imgnode
                else:
                    if not texture:
                        texture = imgnode
            case (ShaderParameterType.FLOAT |
                  ShaderParameterType.FLOAT2 |
                  ShaderParameterType.FLOAT3 |
                  ShaderParameterType.FLOAT4 |
                  ShaderParameterType.FLOAT4X4 |
                  ShaderParameterType.SAMPLER |
                  ShaderParameterType.CBUFFER):
                create_parameter_node(node_tree, param)
            case ShaderParameterType.UNKNOWN:
                continue
            case _:
                raise Exception(f"Unknown shader parameter! {param.type=} {param.name=}")

    use_diff = True if texture else False
    use_diff2 = True if texture2 else False
    use_diff3 = True if texture3 else False
    use_bump = True if bumptex else False
    blend_mode = "OPAQUE"

    is_alpha = True if filename in ShaderManager.alpha_shaders() else False
    if is_alpha:
        blend_mode = "BLEND"

    if use_diff:
        if use_diff3 and texture2 != None and texture3 != None:
            texture = link_rdr1_diffuses_3(b, texture, texture2, texture3)
        elif use_diff2 and (texture2 != None or texture3 != None): # Sometimes a shader with 2 layers will have the sampler3 used instead of sampler2
            texture = link_rdr1_diffuses_2(b, texture, texture2 if texture2 != None else texture3)
        else:
            link_diffuse(b, texture)

    if use_bump:
        link_normal(b, bumptex)
    bsdf.inputs["Specular IOR Level"].default_value = 0

    # link value parameters
    link_value_shader_parameters(b)

    mat.blend_method = blend_mode

def link_rdr1_diffuses_2(b: ShaderBuilder, c0, c1):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links

    tc1 = node_tree.nodes[get_uv_map_name(1)]
    links.new(tc1.outputs["UV"], c1.inputs[0])

    # Lerp: lerp(c0, c1, c1.a)
    lerp1 = node_tree.nodes.new("ShaderNodeMixRGB")
    lerp1.blend_type = 'MIX'
    links.new(c0.outputs["Color"], lerp1.inputs["Color1"])
    links.new(c1.outputs["Color"], lerp1.inputs["Color2"])
    links.new(c1.outputs["Alpha"], lerp1.inputs["Fac"])

    # Create a Value node to set Alpha
    alpha_node = node_tree.nodes.new("ShaderNodeValue")
    alpha_node.outputs[0].default_value = 1.0

    # Connect the output of the Mix RGB node to the input of the Principled BSDF node
    links.new(lerp1.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(alpha_node.outputs[0], bsdf.inputs["Alpha"])

def link_rdr1_diffuses_3(b: ShaderBuilder, c0, c1, c2):
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links

    tc1 = node_tree.nodes[get_uv_map_name(1)]
    tc2 = node_tree.nodes[get_uv_map_name(2)]

    links.new(tc1.outputs["UV"], c1.inputs[0])
    links.new(tc2.outputs["UV"], c2.inputs[0])

    # First lerp: lerp(c0, c1, c1.a)
    lerp1 = node_tree.nodes.new("ShaderNodeMixRGB")
    lerp1.blend_type = 'MIX'
    links.new(c0.outputs["Color"], lerp1.inputs["Color1"])
    links.new(c1.outputs["Color"], lerp1.inputs["Color2"])
    links.new(c1.outputs["Alpha"], lerp1.inputs["Fac"])

    # Second lerp: lerp(lerp(c0, c1, c1.a), c2, c2.a)
    lerp2 = node_tree.nodes.new("ShaderNodeMixRGB")
    lerp2.blend_type = 'MIX'
    links.new(lerp1.outputs["Color"], lerp2.inputs["Color1"])
    links.new(c2.outputs["Color"], lerp2.inputs["Color2"])
    links.new(c2.outputs["Alpha"], lerp1.inputs["Fac"])

    # Create a Value node to set Alpha
    alpha_node = node_tree.nodes.new("ShaderNodeValue")
    alpha_node.outputs[0].default_value = 1.0

    # Connect the output of the final lerp to the Base Color input of the Principled BSDF node
    links.new(lerp2.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(alpha_node.outputs[0], bsdf.inputs["Alpha"])
    node_tree = b.node_tree
    bsdf = b.bsdf
    links = node_tree.links

    # First lerp: lerp(c0, c1, c1.a)
    lerp1 = node_tree.nodes.new("ShaderNodeMixRGB")
    lerp1.blend_type = 'MIX'
    links.new(c0.outputs["Color"], lerp1.inputs["Color1"])
    links.new(c1.outputs["Color"], lerp1.inputs["Color2"])
    links.new(c1.outputs["Alpha"], lerp1.inputs["Fac"])

    # Second lerp: lerp(lerp(c0, c1, c1.a), c2, c2.a)
    lerp2 = node_tree.nodes.new("ShaderNodeMixRGB")
    lerp2.blend_type = 'MIX'
    links.new(lerp1.outputs["Color"], lerp2.inputs["Color1"])
    links.new(c2.outputs["Color"], lerp2.inputs["Color2"])
    links.new(c2.outputs["Alpha"], lerp1.inputs["Fac"])

    # Create a Value node to set Alpha
    alpha_node = node_tree.nodes.new("ShaderNodeValue")
    alpha_node.outputs[0].default_value = 1.0

    # Connect the output of the final lerp to the Base Color input of the Principled BSDF node
    links.new(lerp2.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(alpha_node.outputs[0], bsdf.inputs["Alpha"])

def create_shader(filename: str):
    shader = ShaderManager.find_shader(filename, game=SollumzGame.RDR1)
    
    if shader is None:
        raise AttributeError(f"Shader '{filename}' does not exist!")

    filename = shader.filename  # in case `filename` was hashed initially
    base_name = ShaderManager.find_shader_base_name(filename, game=SollumzGame.RDR1)

    mat = bpy.data.materials.new(filename.replace(".sps", ""))
    mat.sollum_type = MaterialType.SHADER
    mat.use_nodes = True
    mat.shader_properties.name = base_name
    mat.shader_properties.filename = filename

    if isinstance(shader.render_bucket, int):
        bucket_str = bucket_mapping.get(RenderBucket(shader.render_bucket), "OPAQUE")
        mat.shader_properties.renderbucket = bucket_str
    else:
        mat.shader_properties.renderbucket = shader.render_bucket[0]

    bsdf, material_output = find_bsdf_and_material_output(mat)
    assert material_output is not None, "ShaderNodeOutputMaterial not found in default node_tree!"
    assert bsdf is not None, "ShaderNodeBsdfPrincipled not found in default node_tree!"

    builder = ShaderBuilder(shader=shader,
                            filename=filename,
                            material=mat,
                            node_tree=mat.node_tree,
                            material_output=material_output,
                            bsdf=bsdf)

    create_uv_map_nodes(builder)

    RDR_create_basic_shader_nodes(builder, game=SollumzGame.RDR1)

    link_uv_map_nodes_to_textures(builder)

    organize_node_tree(builder)

    return mat