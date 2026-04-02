"""
Apply a rotation (X/Y/Z Euler angles) around the WORLD ORIGIN to an articulated 
USD robot. This modifies the CURRENTLY OPENED stage in Isaac Sim.

What gets rotated and why
─────────────────────────
• Top-level ArticulationRoots, RigidBodies, and Joint Xforms.
  By revolving the highest-level physics containers around the world origin, 
  the entire robot is rotated perfectly in world space, exactly as requested.
  
• Because all rigid bodies undergo the exact same world-origin rotation, 
  their relative distances and local coordinate systems remain mathematically 
  identical. Therefore, joint connection frames (localPos/localRot) are 
  natively preserved and do NOT need to be modified.

Usage:
1. Open your Rover USD in Isaac Sim.
2. Ensure the rover is positioned where you want it relative to the origin 
   (typically you want the robot centered at 0,0,0 before baking a rotation).
3. Open Window > Script Editor.
4. Adjust the Configuration block below so +X becomes the forward axis.
5. Run the script.
6. Save the file (File > Save).
"""

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# ── Configuration ─────────────────────────────────────────────────────────────
# Aligning Rovers to +X Forward:
# • If the rover currently faces +Y forward, set ROT_Z = -90.0
# • If the rover currently faces -Y forward, set ROT_Z = 90.0
# • If the rover currently faces +Z forward, set ROT_Y = 90.0
# • If the rover currently faces -Z forward, set ROT_Y = -90.0

ROT_X = 0.0
ROT_Y = 0.0
ROT_Z = -90.0  # Default correction for a +Y forward rover
# ──────────────────────────────────────────────────────────────────────────────


def build_world_rotation_matrix(rot_x: float, rot_y: float, rot_z: float) -> Gf.Matrix4d:
    """Compose a 4x4 rotation matrix from X→Y→Z Euler angles (degrees)."""
    rot = Gf.Rotation(Gf.Vec3d(1, 0, 0), rot_x) * \
          Gf.Rotation(Gf.Vec3d(0, 1, 0), rot_y) * \
          Gf.Rotation(Gf.Vec3d(0, 0, 1), rot_z)
    
    # Return a 4x4 matrix with pure rotation (no translation)
    return Gf.Matrix4d(rot, Gf.Vec3d(0, 0, 0))


def get_top_level_physics_prims(stage: Usd.Stage):
    """
    Finds all top-level physics prims (Articulation Root, RigidBody, or Joint).
    'Top-level' means they do not have a parent/ancestor that is ALSO in this list. 
    Rotating just these guarantees all nested sub-components are safely carried along.
    """
    targets =[]
    for prim in stage.Traverse():
        is_art_root = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        is_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_joint = prim.GetAttribute("physics:localPos0").IsValid() or prim.IsA(UsdPhysics.Joint)
        
        if is_art_root or is_rb or is_joint:
            targets.append(prim)
    
    top_level =[]
    for prim in targets:
        parent = prim.GetParent()
        has_target_ancestor = False
        
        # Walk up the tree to see if an ancestor is also in our target list
        while parent and parent.IsValid() and not parent.IsPseudoRoot():
            if parent in targets:
                has_target_ancestor = True
                break
            parent = parent.GetParent()
        
        if not has_target_ancestor:
            top_level.append(prim)
            
    return top_level


def apply_world_rotation(rot_x: float, rot_y: float, rot_z: float) -> None:
    stage = omni.usd.get_context().get_stage()
    
    if not stage:
        print("[ERROR] No USD stage is currently open in Isaac Sim.")
        return

    rot_mat = build_world_rotation_matrix(rot_x, rot_y, rot_z)
    top_level_prims = get_top_level_physics_prims(stage)

    if not top_level_prims:
        print("[WARN] No ArticulationRoot, RigidBody, or Joint prims found.")
        return

    # Use XformCache to accurately read current world transforms
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # Wrap in an undo block so you can Ctrl+Z in Isaac Sim if the rotation is wrong
    with omni.kit.undo.group():
        for prim in top_level_prims:
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                continue
            
            # 1. Get the current world transform
            world_transform = xform_cache.GetLocalToWorldTransform(prim)
            
            # 2. Revolve/Rotate around the world origin
            # In USD (row-vector math), multiplying the world matrix by a rotation matrix 
            # applies that rotation globally around 0,0,0.
            new_world_transform = world_transform * rot_mat
            
            # 3. Compute the new local transform to maintain safe parent/child hierarchy
            parent = prim.GetParent()
            if parent and not parent.IsPseudoRoot():
                parent_world = xform_cache.GetLocalToWorldTransform(parent)
            else:
                parent_world = Gf.Matrix4d(1.0)
            
            # Local = World * Parent_World_Inverse
            parent_world_inv = parent_world.GetInverse()
            new_local_transform = new_world_transform * parent_world_inv
            
            # 4. Decompose the matrix into standard Translation, Rotation, and Scale
            decomposed = Gf.Transform(new_local_transform)
            
            # 5. Apply standard operations. Clearing first ensures custom/weird pivots 
            # from URDF imports don't corrupt the rotation math.
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetTranslation())
            xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetRotation().GetQuat())
            xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetScale())
            
            print(f"[Rotated] {prim.GetPath()}")

    print(
        f"\nDone applying world-origin rotation to active stage.\n"
        f"  Total root objects rotated: {len(top_level_prims)}\n"
        f"  Rotation Applied: X={rot_x}°  Y={rot_y}°  Z={rot_z}°\n\n"
        f"You can now inspect the changes. If correct, go to File > Save."
    )

# ── Entry point ───────────────────────────────────────────────────────────────
apply_world_rotation(ROT_X, ROT_Y, ROT_Z)
