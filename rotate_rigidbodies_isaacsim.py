"""
Apply a rotation (X/Y/Z Euler angles) to an articulated USD robot, keeping
physics and visuals consistent.

What gets rotated and why
─────────────────────────
1. Xformable children of ROOT rigid body prims only (those with no rigid body ancestor)
   → visuals/collision shapes of the main body live here; PhysX preserves them.
   → Children of non-root rigid bodies (e.g. thruster link cylinders) are intentionally
     skipped — their placement is already handled by rotating the joint frames below.
     Rotating both would double-rotate them.

2. Joint local frames (localPos0/1, localRot0/1) for every joint
   → these express the joint attachment point in the parent/child body frame.
     Rotating these moves thruster links (and any other child links) to their
     correct rotated positions.

What is intentionally NOT rotated
──────────────────────────────────
• Rigid body prims themselves — PhysX overwrites their world transforms on
  sim init; editing their xformOps has no effect.
• The ArticulationRoot xform — use init_state.rot in the Isaac Lab config
  to control spawn/reset placement.

Run inside the Isaac Sim Script Editor (Window > Script Editor).
Adjust the Configuration block at the top as needed.
"""

import math

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_PATH  = "C:\\IsaacLab\\source\\isaaclab_assets\\data\\Robots\\SpaceR-TheDreamLab\\UniluFP\\ultra_flat_col.usd"
OUTPUT_PATH = "C:\\IsaacLab\\source\\isaaclab_assets\\data\\Robots\\SpaceR-TheDreamLab\\UniluFP\\2deep_ultra_flat_col.usd"

# Rotation in degrees applied around each axis (order: X → Y → Z)
ROT_X = 0.0
ROT_Y = 0.0
ROT_Z = 45.0

# Translation offset (same units as the USD stage)
TRANS_X = 0.0
TRANS_Y = 0.0
TRANS_Z = 0.0
# ──────────────────────────────────────────────────────────────────────────────


def build_rotation_quat(rot_x: float, rot_y: float, rot_z: float) -> Gf.Quatd:
    """Compose a quaternion from X→Y→Z Euler angles (degrees)."""
    def axis_quat(degrees: float, axis: Gf.Vec3d) -> Gf.Quatd:
        half = math.radians(degrees) / 2.0
        return Gf.Quatd(math.cos(half), axis * math.sin(half))

    qx = axis_quat(rot_x, Gf.Vec3d(1, 0, 0))
    qy = axis_quat(rot_y, Gf.Vec3d(0, 1, 0))
    qz = axis_quat(rot_z, Gf.Vec3d(0, 0, 1))
    return qx * qy * qz


def build_transform_matrix(rot_quat: Gf.Quatd, trans_x: float, trans_y: float, trans_z: float) -> Gf.Matrix4d:
    rot_mat = Gf.Matrix4d()
    rot_mat.SetRotate(rot_quat)
    trans_mat = Gf.Matrix4d()
    trans_mat.SetTranslate(Gf.Vec3d(trans_x, trans_y, trans_z))
    return rot_mat * trans_mat


def has_rigidbody_ancestor(prim: Usd.Prim) -> bool:
    parent = prim.GetParent()
    while parent and parent.IsValid() and not parent.IsPseudoRoot():
        if parent.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        parent = parent.GetParent()
    return False


def is_articulation_root_child(prim: Usd.Prim) -> bool:
    """True if the prim's direct parent has ArticulationRootAPI."""
    parent = prim.GetParent()
    return parent and parent.IsValid() and parent.HasAPI(UsdPhysics.ArticulationRootAPI)


def rotate_xformable_children(rb_prims, rot_quat: Gf.Quatd) -> int:
    """Rotate the Xformable direct children of the articulation root's direct child rigid bodies.

    We only touch rigid bodies that are DIRECT children of the ArticulationRoot prim
    (i.e., the primary body / base_link). All other rigid bodies (arms, thrusters, etc.)
    are skipped — their placement is handled by rotating joint frames, and their
    children live in their own local frame which doesn't need changing.

    The rotation is applied IN LOCAL SPACE (parent rigid body frame), not world space.
    A world-space roundtrip introduces spurious translations for rigid bodies that are
    not at the world origin.
    """
    rot_mat = Gf.Matrix4d()
    rot_mat.SetRotate(rot_quat)

    # Snapshot local transforms before any edits
    prim_data = []
    for rb in rb_prims:
        if not is_articulation_root_child(rb):
            continue
        for child in rb.GetChildren():
            xformable = UsdGeom.Xformable(child)
            if not xformable:
                continue
            local_mat = xformable.GetLocalTransformation(Usd.TimeCode.Default())
            prim_data.append((child, local_mat))

    if not prim_data:
        print("[WARN] No Xformable children of articulation-root-child rigid bodies found.")
        return 0

    for prim, local_mat in prim_data:
        # Apply rotation in the parent rigid body's local frame:
        # new_local = rot_mat * local_mat
        # This rotates both the local position vector and the local orientation.
        new_local = rot_mat * local_mat

        decomposed = Gf.Transform(new_local)
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetTranslation())
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetRotation().GetQuat())
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(decomposed.GetScale())
        print(f"  [visual] {prim.GetPath()}")

    return len(prim_data)


def rotate_joint_frames(stage: Usd.Stage, rot_quat: Gf.Quatd, root_rb_paths: set) -> int:
    """
    Rotate joint local frames selectively based on whether each side's body was visually rotated.

    A joint has two sides:
      - body0 (parent): localPos0 / localRot0 expressed in body0's local frame
      - body1 (child):  localPos1 / localRot1 expressed in body1's local frame

    We only rotated the visual children of ROOT rigid bodies (those with no RB ancestor).
    So we only rotate a joint's local frame on a given side if that side's body is a root RB.
    Rotating non-root sides would move arm/thruster attachment points in their own local frame,
    which was never changed — causing the double-rotation seen with Pingu's arms.

    Joints are detected by physics:localPos0 (UsdPhysics.JointAPI does not exist in this version).
    """
    rotation = Gf.Rotation(rot_quat)

    def get_body_path(prim: Usd.Prim, rel_name: str) -> str | None:
        rel = prim.GetRelationship(rel_name)
        targets = rel.GetTargets() if rel else []
        return str(targets[0]) if targets else None

    joint_count = 0
    for prim in stage.Traverse():
        if not prim.GetAttribute("physics:localPos0"):
            continue

        body0_path = get_body_path(prim, "physics:body0")
        body1_path = get_body_path(prim, "physics:body1")

        rotate_side0 = body0_path in root_rb_paths or body0_path is None
        rotate_side1 = body1_path in root_rb_paths or body1_path is None

        if not rotate_side0 and not rotate_side1:
            print(f"  [joint skip] {prim.GetPath()} (neither body is root RB)")
            continue

        if rotate_side0:
            pos0_attr = prim.GetAttribute("physics:localPos0")
            if pos0_attr.HasAuthoredValue():
                pos0_attr.Set(Gf.Vec3f(rotation.TransformDir(Gf.Vec3d(pos0_attr.Get()))))
            rot0_attr = prim.GetAttribute("physics:localRot0")
            if rot0_attr and rot0_attr.HasAuthoredValue():
                rot0_attr.Set(Gf.Quatf(rot_quat * Gf.Quatd(rot0_attr.Get())))

        if rotate_side1:
            pos1_attr = prim.GetAttribute("physics:localPos1")
            if pos1_attr and pos1_attr.HasAuthoredValue():
                pos1_attr.Set(Gf.Vec3f(rotation.TransformDir(Gf.Vec3d(pos1_attr.Get()))))
            rot1_attr = prim.GetAttribute("physics:localRot1")
            if rot1_attr and rot1_attr.HasAuthoredValue():
                rot1_attr.Set(Gf.Quatf(rot_quat * Gf.Quatd(rot1_attr.Get())))

        print(f"  [joint]  {prim.GetPath()}")
        joint_count += 1

    return joint_count



def apply_transform(input_path: str, output_path: str,
                    rot_x: float, rot_y: float, rot_z: float,
                    trans_x: float, trans_y: float, trans_z: float) -> None:
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print(f"[ERROR] Could not open: {input_path}")
        return

    rot_quat = build_rotation_quat(rot_x, rot_y, rot_z)
    world_delta = build_transform_matrix(rot_quat, trans_x, trans_y, trans_z)

    rb_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not rb_prims:
        print("[WARN] No rigid body prims found — nothing to transform.")
        return

    root_rb_paths = {
        str(prim.GetPath()) for prim in rb_prims if not has_rigidbody_ancestor(prim)
    }
    print(f"  Root rigid bodies: {root_rb_paths}")

    n_visual = rotate_xformable_children(rb_prims, rot_quat)
    n_joints = rotate_joint_frames(stage, rot_quat, root_rb_paths)

    print(
        f"\nDone.\n"
        f"  Visual prims rotated : {n_visual}\n"
        f"  Joint frames rotated : {n_joints}\n"
        f"  Rotation  X={rot_x}°  Y={rot_y}°  Z={rot_z}°\n"
        f"  Translation  X={trans_x}  Y={trans_y}  Z={trans_z}"
    )

    stage.Export(output_path)
    print(f"Saved: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
apply_transform(INPUT_PATH, OUTPUT_PATH, ROT_X, ROT_Y, ROT_Z, TRANS_X, TRANS_Y, TRANS_Z)
