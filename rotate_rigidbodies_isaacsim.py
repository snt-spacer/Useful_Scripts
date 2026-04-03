"""
Apply a permanent global transformation for Isaac Lab (Root Deformation Baking).

Because Isaac Lab forces the Root Rigid Body to [1,0,0,0] at runtime, any rotation 
applied directly to the Root Rigid Body is erased. This script solves this by:
1. Forcing the Root Rigid Body to Identity.
2. Baking the rotation into the Visuals/Collisions INSIDE the Root Rigid Body.
3. Rotating the joint anchor points (localPos/localRot) connected to the Root.
4. Revolving all child rigid bodies around the origin to match.
5. Cleans up the USD by physically removing old `rotateXYZ` properties.

Usage:
1. Open your Rover USD in Isaac Sim.
2. Open Window > Script Editor.
3. Run the script.
4. Save the file.
"""

import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Sdf

# ── Configuration ─────────────────────────────────────────────────────────────
ROT_X = 0.0
ROT_Y = 0.0
ROT_Z = -90.0  # Correction to make +Y facing rovers face +X forward

TRANS_X = 0.0
TRANS_Y = 0.0
TRANS_Z = 0.0
# ──────────────────────────────────────────────────────────────────────────────


def get_global_transform_matrix(rot_x: float, rot_y: float, rot_z: float, 
                                trans_x: float, trans_y: float, trans_z: float) -> Gf.Matrix4d:
    """Compose a 4x4 matrix for Rotation followed by Translation."""
    rot = Gf.Rotation(Gf.Vec3d(1, 0, 0), rot_x) * \
          Gf.Rotation(Gf.Vec3d(0, 1, 0), rot_y) * \
          Gf.Rotation(Gf.Vec3d(0, 0, 1), rot_z)
    
    rot_mat = Gf.Matrix4d(rot, Gf.Vec3d(0, 0, 0))
    trans_mat = Gf.Matrix4d().SetTranslate(Gf.Vec3d(trans_x, trans_y, trans_z))
    return rot_mat * trans_mat


def set_xform_ops(prim: Usd.Prim, matrix: Gf.Matrix4d):
    """Safely apply standard Isaac Sim xformOps and physically clean up old ones."""
    # 1. Strip away legacy orientation properties (like rotateXYZ) to prevent USDA clutter
    for prop in prim.GetAuthoredProperties():
        name = prop.GetName()
        if name.startswith("xformOp:rotate") or name.startswith("xformOp:orient") or name.startswith("xformOp:transform"):
            prim.RemoveProperty(name)

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    decomp = Gf.Transform(matrix)
    
    # 2. Translate (double3)
    trans = decomp.GetTranslation()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(trans))
    
    # 3. Orient (quatd) - This is the ONLY rotation component that will be preserved
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(decomp.GetRotation().GetQuat())
    
    # 4. Scale (float3)
    scale = decomp.GetScale()
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(scale))


def apply_isaac_lab_bake():
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("[ERROR] No USD stage is currently open.")
        return

    # 1. Identify the Root Rigid Body (Base Link)
    joints =[j for j in stage.Traverse() if j.IsA(UsdPhysics.Joint) or j.GetAttribute("physics:localPos0")]
    body1_targets = set()
    for j in joints:
        rel = j.GetRelationship("physics:body1")
        if rel and rel.GetTargets():
            body1_targets.add(str(rel.GetTargets()[0]))
            
    rbs =[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    root_rb = None
    for rb in rbs:
        # The true base link is the rigid body that has NO incoming joints (never body1)
        if str(rb.GetPath()) not in body1_targets:
            root_rb = rb
            break
            
    if not root_rb:
        print("[ERROR] Could not identify the Root Rigid Body. Ensure physics relationships are valid.")
        return
        
    print(f"[INFO] Identified Root Rigid Body: {root_rb.GetPath()}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    M_global = get_global_transform_matrix(ROT_X, ROT_Y, ROT_Z, TRANS_X, TRANS_Y, TRANS_Z)
    
    # We will force root_rb to be Identity relative to its parent.
    root_W_old = xform_cache.GetLocalToWorldTransform(root_rb)
    parent = root_rb.GetParent()
    parent_W = xform_cache.GetLocalToWorldTransform(parent) if parent else Gf.Matrix4d(1.0)
    W_root_new = parent_W 
    
    child_L_new = {}
    joint_updates =[]
    rb_L_new = {}

    # --- PRE-CALCULATE ALL MATHEMATICS BEFORE EDITING USD ---

    # A. Visuals & Collisions INSIDE the Root Rigid Body
    for child in root_rb.GetChildren():
        if child.IsA(UsdGeom.Xformable):
            W_child_old = xform_cache.GetLocalToWorldTransform(child)
            W_child_new = W_child_old * M_global
            child_L_new[child.GetPath()] = W_child_new * W_root_new.GetInverse()

    # B. Joint Anchors attached to the Root Rigid Body
    for j in joints:
        for side in ['0', '1']:
            rel = j.GetRelationship(f"physics:body{side}")
            if rel and rel.GetTargets() and str(rel.GetTargets()[0]) == str(root_rb.GetPath()):
                pos_attr = j.GetAttribute(f"physics:localPos{side}")
                rot_attr = j.GetAttribute(f"physics:localRot{side}")
                
                pos = pos_attr.Get() if (pos_attr and pos_attr.IsValid()) else Gf.Vec3f(0.0)
                rot = rot_attr.Get() if (rot_attr and rot_attr.IsValid()) else Gf.Quatf(1.0)
                
                # Transform the joint anchor to global, rotate it, bring it back to new local
                L_anchor_old = Gf.Matrix4d(Gf.Rotation(Gf.Quatd(rot)), Gf.Vec3d(pos))
                W_anchor_old = L_anchor_old * root_W_old
                W_anchor_new = W_anchor_old * M_global
                L_anchor_new = W_anchor_new * W_root_new.GetInverse()
                
                decomp = Gf.Transform(L_anchor_new)
                joint_updates.append((j, side, decomp.GetTranslation(), decomp.GetRotation().GetQuat()))

    # C. All OTHER Rigid Bodies (Child Links)
    other_rbs =[rb for rb in rbs if rb != root_rb]
    for rb in other_rbs:
        W_rb_old = xform_cache.GetLocalToWorldTransform(rb)
        W_rb_new = W_rb_old * M_global
        
        p_rb = rb.GetParent()
        p_W_old = xform_cache.GetLocalToWorldTransform(p_rb) if p_rb else Gf.Matrix4d(1.0)
        rb_L_new[rb.GetPath()] = W_rb_new * p_W_old.GetInverse()


    # --- APPLY ALL MATHEMATICS SAFELY ---
    with omni.kit.undo.group():
        
        # 1. Update geometry inside the root link
        for path, L_mat in child_L_new.items():
            set_xform_ops(stage.GetPrimAtPath(path), L_mat)
            print(f"[Rotated Visuals] {path}")
            
        # 2. Update child rigid bodies
        for path, L_mat in rb_L_new.items():
            set_xform_ops(stage.GetPrimAtPath(path), L_mat)
            print(f"[Revolved Link] {path}")
            
        # 3. Update Joint Anchors attached to Root
        for j, side, new_pos, new_rot in joint_updates:
            # Enforce strict float precision for joints
            pos_attr = j.GetAttribute(f"physics:localPos{side}")
            if not pos_attr:
                pos_attr = j.CreateAttribute(f"physics:localPos{side}", Sdf.ValueTypeNames.Point3f)
            pos_attr.Set(Gf.Vec3f(new_pos))
            
            rot_attr = j.GetAttribute(f"physics:localRot{side}")
            if not rot_attr:
                rot_attr = j.CreateAttribute(f"physics:localRot{side}", Sdf.ValueTypeNames.Quatf)
            rot_attr.Set(Gf.Quatf(new_rot))
            print(f"[Rotated Joint Anchor] {j.GetPath()} (side {side})")

        # 4. FORCE the Root Rigid Body to Identity (Zeroing it out for Isaac Lab)
        set_xform_ops(root_rb, Gf.Matrix4d(1.0))
        print(f"[Zeroed Root Link] {root_rb.GetPath()}")

    print("\n--- DONE ---")
    print("The robot's default orientation is perfectly baked for Isaac Lab.")
    print("You can now safely spawn this robot with an Identity state[1, 0, 0, 0] and it will face +X.")

# ── Entry point ───────────────────────────────────────────────────────────────
apply_isaac_lab_bake()
