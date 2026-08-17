"""
RoArm Tabletop RGB-D Dataset Collection for RealDepth.

A fixed-viewpoint tabletop scenario:
  * RoArm-M2 robot (URDF imported from waveshare roarm_description), base fixed
    at the origin, recolored dark gray, joints driven through a smooth random
    trajectory within each episode (temporal motion for the ConvGRU model).
  * Small colored cubes (2-5 cm) scattered on the surface within the arm's reach,
    random colors / poses / count per episode.
  * A single FIXED camera (e.g. 20 cm height) looking at the workspace -- same
    extrinsics every frame, mimicking a mounted workstation camera.
  * Random surface material per episode.

Each episode is one temporal sequence (camera + cube layout fixed within it, the
arm moves). A frame-number gap is inserted between episodes so SequenceDataset /
split_dataset treat each episode as a separate sequence.

Usage (run from Isaac Sim's python, with EULA accepted):
    OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src \
    /home/rassul_pc/isaac/venv/bin/python scripts/collect_roarm.py \
        --num_episodes 4 --frames_per_episode 75 --headless \
        --output_dir collected_dataset/roarm_test
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---- Parse args BEFORE SimulationApp (it consumes some args) ----
parser = argparse.ArgumentParser(description="Collect RoArm tabletop RGB-D dataset")
parser.add_argument("--urdf_path", type=str,
                    default="assets/roarm_description/urdf/roarm_description.urdf",
                    help="Path to the RoArm URDF (with relative mesh paths)")
parser.add_argument("--robot_usd", type=str,
                    default="assets/roarm_usd/roarm.usd",
                    help="Cached USD the URDF is imported to (created on first run)")
parser.add_argument("--output_dir", type=str, default="",
                    help="Output directory (default: collected_dataset/roarm_<timestamp>)")
parser.add_argument("--num_episodes", type=int, default=50,
                    help="Number of episodes (= temporal sequences) to collect")
parser.add_argument("--frames_per_episode", type=int, default=100,
                    help="Frames captured per episode")
parser.add_argument("--width", type=int, default=640, help="Image width")
parser.add_argument("--height", type=int, default=480, help="Image height")
parser.add_argument("--fov_h", type=float, default=65.0,
                    help="Fixed horizontal FOV in degrees")
parser.add_argument("--fps", type=int, default=30, help="Simulation step rate")
parser.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
parser.add_argument("--max_depth_mm", type=int, default=3000,
                    help="Max depth clamp in mm (default 3000 = 3 m)")
parser.add_argument("--scene_gap", type=int, default=50,
                    help="Frame-number gap inserted between episodes")
parser.add_argument("--min_cubes", type=int, default=3, help="Min cubes per episode")
parser.add_argument("--max_cubes", type=int, default=8, help="Max cubes per episode")
# Fixed camera pose (world frame, metres). Looks at --cam_target.
parser.add_argument("--cam_pos", type=float, nargs=3, default=[0.30, -0.32, 0.26],
                    help="Fixed camera position x y z (3/4 view, ~26 cm height)")
parser.add_argument("--cam_target", type=float, nargs=3, default=[0.05, -0.03, 0.06],
                    help="Point the fixed camera looks at (workspace centre)")
parser.add_argument("--seed", type=int, default=0, help="RNG seed")
args, unknown = parser.parse_known_args()

# ---- Launch Isaac Sim ----
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# Now safe to import Omniverse / Isaac Sim modules
import omni.kit.commands
import omni.usd as omni_usd
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from isaacsim.asset.importer.urdf import _urdf
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

import cv2


def log(msg):
    sys.stderr.write(f"[RoArm] {msg}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# URDF -> USD import (cached)
# ---------------------------------------------------------------------------
def import_robot_usd(urdf_path, usd_path):
    """Import the URDF to a cached USD asset (one-time) and return its path.

    Uses a file-based import with make_default_prim=True so the mesh sub-layers
    are written and resolvable, and so the asset can be referenced by its default
    prim. Reuses the cached USD on subsequent runs.
    """
    usd_path = Path(usd_path).resolve()
    if usd_path.exists():
        log(f"Using cached robot USD: {usd_path}")
        return str(usd_path)

    urdf_path = str(Path(urdf_path).resolve())
    log(f"Importing URDF -> USD (one-time): {urdf_path}")

    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.set_fix_base(True)               # base anchored at a fixed place
    cfg.set_merge_fixed_joints(False)
    cfg.set_make_default_prim(True)      # so we can reference it by default prim
    cfg.set_import_inertia_tensor(True)
    cfg.set_distance_scale(1.0)          # URDF is already in metres
    cfg.set_density(0.0)
    cfg.set_self_collision(False)
    cfg.set_collision_from_visuals(False)
    cfg.set_create_physics_scene(False)  # World provides the physics scene
    # Position drives so the arm holds / tracks joint-position targets.
    try:
        cfg.set_default_drive_type(_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION)
    except Exception as e:
        log(f"drive-type set skipped: {e}")
    try:
        cfg.set_default_drive_strength(800.0)          # stiffness
        cfg.set_default_position_drive_damping(60.0)   # damping
    except Exception as e:
        log(f"drive-strength set skipped: {e}")

    usd_path.parent.mkdir(parents=True, exist_ok=True)
    status, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=cfg,
        dest_path=str(usd_path),
    )
    log(f"Imported robot USD: {usd_path} (status={status})")
    return str(usd_path)


def deinstance_subtree(stage, root_path):
    """Disable instanceable on a subtree so its (otherwise prototype-hidden)
    visual meshes become regular, renderable prims and accept material
    overrides. Without this the imported robot's meshes do not render."""
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if p.IsInstanceable():
            p.SetInstanceable(False)
            n += 1
    log(f"De-instanced {n} prims under {root_path}")


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
def _make_preview_material(stage, mat_path, color, roughness=0.6, metallic=0.0):
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind_material(stage, prim_path, mat):
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(mat)


def recolor_robot_dark_gray(stage, robot_prim_path):
    """Force the whole robot dark gray by binding a material at its root with
    'strongerThanDescendants' strength, which overrides the per-link bindings
    the URDF importer created."""
    mat = _make_preview_material(
        stage, "/World/Materials/RoArmDarkGray",
        color=(0.18, 0.18, 0.20), roughness=0.55, metallic=0.15)
    root = stage.GetPrimAtPath(robot_prim_path)
    UsdShade.MaterialBindingAPI(root).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    log(f"Bound dark-gray material at {robot_prim_path} (stronger than descendants)")


# ---------------------------------------------------------------------------
# Lighting
# ---------------------------------------------------------------------------
_light_prims = []


def setup_lights(stage):
    global _light_prims
    _light_prims = []

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1000.0)
    _light_prims.append(("dome", "/World/Lights/Dome"))

    dist = UsdLux.DistantLight.Define(stage, "/World/Lights/Distant")
    dist.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(stage.GetPrimAtPath("/World/Lights/Distant")) \
        .AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))
    _light_prims.append(("distant", "/World/Lights/Distant"))

    for i in range(3):
        p = f"/World/Lights/Point_{i}"
        sl = UsdLux.SphereLight.Define(stage, p)
        sl.CreateIntensityAttr(3000.0)
        sl.CreateRadiusAttr(0.05)
        UsdGeom.Xformable(stage.GetPrimAtPath(p)).AddTranslateOp().Set(
            Gf.Vec3f(0.0, 0.0, 1.0))
        _light_prims.append(("point", p))


def randomize_lighting(stage, rng):
    for ltype, lpath in _light_prims:
        prim = stage.GetPrimAtPath(lpath)
        if not prim.IsValid():
            continue
        if ltype == "dome":
            light = UsdLux.DomeLight(prim)
            light.GetIntensityAttr().Set(float(rng.uniform(300, 2500)))
            c = rng.uniform(0.75, 1.0, size=3)
            light.GetColorAttr().Set(Gf.Vec3f(*[float(x) for x in c]))
        elif ltype == "distant":
            light = UsdLux.DistantLight(prim)
            light.GetIntensityAttr().Set(float(rng.uniform(800, 5000)))
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddRotateXYZOp().Set(Gf.Vec3f(
                float(rng.uniform(-70, -20)), float(rng.uniform(-180, 180)), 0.0))
        elif ltype == "point":
            light = UsdLux.SphereLight(prim)
            light.GetIntensityAttr().Set(float(rng.uniform(0, 4000)))
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3f(
                float(rng.uniform(-0.5, 0.5)),
                float(rng.uniform(-0.7, 0.2)),
                float(rng.uniform(0.4, 1.2))))


# ---------------------------------------------------------------------------
# Table surface + cubes
# ---------------------------------------------------------------------------
def _set_xform(prim, translate, scale, rotate_z_deg=0.0):
    """Idempotently set translate/rotate/scale via XformCommonAPI (no duplicate
    xform-op errors when called repeatedly across episodes)."""
    api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(*[float(v) for v in translate]))
    api.SetRotate(Gf.Vec3f(0.0, 0.0, float(rotate_z_deg)),
                  UsdGeom.XformCommonAPI.RotationOrderXYZ)
    api.SetScale(Gf.Vec3f(*[float(v) for v in scale]))


def build_table(stage):
    """A large static box whose top sits at z=0; random material per episode."""
    path = "/World/Table"
    UsdGeom.Cube.Define(stage, path)
    # 4 x 4 x 0.1 m, top face at z=0 (cube default edge = 2 m -> scale halves it)
    _set_xform(stage.GetPrimAtPath(path), (0.0, 0.0, -0.05), (0.5, 0.5, 0.05))
    return path


def randomize_table_material(stage, rng, idx):
    color = rng.uniform(0.15, 0.85, size=3)
    mat = _make_preview_material(
        stage, f"/World/Materials/Table_{idx:04d}", color=color,
        roughness=float(rng.uniform(0.3, 1.0)),
        metallic=float(rng.choice([0.0, 0.0, 0.0, 0.4])))
    _bind_material(stage, "/World/Table", mat)


# Vivid cube colors
CUBE_COLORS = [
    (0.85, 0.10, 0.10), (0.10, 0.55, 0.85), (0.15, 0.70, 0.20),
    (0.95, 0.75, 0.10), (0.65, 0.15, 0.75), (0.95, 0.45, 0.10),
    (0.10, 0.75, 0.70), (0.90, 0.20, 0.55),
]


def build_cube_pool(stage, max_cubes):
    """Static visual cubes (no physics): we place them exactly on the table top,
    so no settling is needed and USD transforms control them directly."""
    paths = []
    for i in range(max_cubes):
        path = f"/World/Cubes/Cube_{i:02d}"
        UsdGeom.Cube.Define(stage, path)
        paths.append(path)
    return paths


def randomize_cubes(stage, cube_paths, rng, ep_idx):
    """Activate k cubes resting on the table within reach; hide the rest."""
    k = int(rng.integers(args.min_cubes, args.max_cubes + 1))
    order = rng.permutation(len(cube_paths))
    for n, ci in enumerate(order):
        path = cube_paths[ci]
        prim = stage.GetPrimAtPath(path)
        imageable = UsdGeom.Imageable(prim)
        if n < k:
            edge = float(rng.uniform(0.02, 0.05))            # 2-5 cm
            # place within an annulus around the base, resting on the table
            r = float(rng.uniform(0.12, 0.34))
            theta = float(rng.uniform(0, 2 * math.pi))
            x, y = r * math.cos(theta), r * math.sin(theta)
            z = edge / 2.0                                   # sits on z=0 top
            _set_xform(prim, (x, y, z),
                       (edge / 2.0, edge / 2.0, edge / 2.0),
                       rotate_z_deg=float(rng.uniform(0, 360)))
            color = CUBE_COLORS[ci % len(CUBE_COLORS)]
            jitter = rng.uniform(-0.08, 0.08, size=3)
            color = tuple(float(np.clip(c + j, 0.05, 1.0))
                          for c, j in zip(color, jitter))
            mat = _make_preview_material(
                stage, f"/World/Materials/Cube_{ep_idx:04d}_{ci:02d}",
                color=color, roughness=float(rng.uniform(0.2, 0.8)))
            _bind_material(stage, path, mat)
            imageable.MakeVisible()
        else:
            _set_xform(prim, (0.0, 0.0, -5.0), (0.01, 0.01, 0.01))  # hide
            imageable.MakeInvisible()
    return k


# ---------------------------------------------------------------------------
# Fixed camera pose (look-at), built from a rotation matrix so it is robust
# for any viewing direction. Isaac "world" camera axes: +X forward, +Z up.
# ---------------------------------------------------------------------------
def _mat_to_quat(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


def look_at_quat(cam_pos, target):
    f = np.array(target, float) - np.array(cam_pos, float)
    n = np.linalg.norm(f)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    f = f / n                                    # +X forward
    up_ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(f, up_ref))) > 0.999:    # near-vertical: pick a different up
        up_ref = np.array([0.0, 1.0, 0.0])
    l = np.cross(up_ref, f); l = l / np.linalg.norm(l)  # +Y left
    u = np.cross(f, l)                                  # +Z up
    return _mat_to_quat(np.column_stack([f, l, u]))


def set_camera_fov(stage, prim_path, fov_h_deg, width, height):
    cam = UsdGeom.Camera(stage.GetPrimAtPath(prim_path))
    focal = 24.0
    h_aperture = 2.0 * focal * math.tan(math.radians(fov_h_deg) / 2.0)
    v_aperture = h_aperture * (height / float(width))  # square pixels
    cam.GetFocalLengthAttr().Set(focal)
    cam.GetHorizontalApertureAttr().Set(h_aperture)
    cam.GetVerticalApertureAttr().Set(v_aperture)
    # Near clip must be small: the default 1 m would clip the whole tabletop.
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1.0e6))


def read_intrinsics(camera, width, height, fov_h_deg):
    try:
        m = camera.get_intrinsics_matrix()
        fx, fy = float(m[0, 0]), float(m[1, 1])
        cx, cy = float(m[0, 2]), float(m[1, 2])
    except Exception:
        fx = fy = width / (2.0 * math.tan(math.radians(fov_h_deg) / 2.0))
        cx, cy = width / 2.0, height / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": int(width), "height": int(height)}


# ---------------------------------------------------------------------------
# Arm trajectory: smooth per-joint sinusoids within (reduced) joint limits.
# ---------------------------------------------------------------------------
# (lower, upper) usable ranges, slightly inside the URDF hard limits.
ARM_LIMITS = {
    "base_link_to_link1": (-2.6, 2.6),
    "link1_to_link2": (-1.3, 1.3),
    "link2_to_link3": (-0.7, 2.6),
    "link3_to_gripper_link": (0.0, 1.4),
}


class ArmTrajectory:
    def __init__(self, dof_names, rng):
        self.dof_names = dof_names
        self.center = np.zeros(len(dof_names))
        self.amp = np.zeros(len(dof_names))
        self.freq = np.zeros(len(dof_names))
        self.phase = np.zeros(len(dof_names))
        self.lo = np.zeros(len(dof_names))
        self.hi = np.zeros(len(dof_names))
        for i, name in enumerate(dof_names):
            lo, hi = ARM_LIMITS.get(name, (-1.0, 1.0))
            self.lo[i], self.hi[i] = lo, hi
            mid = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            self.center[i] = mid + rng.uniform(-0.3, 0.3) * half
            self.amp[i] = rng.uniform(0.25, 0.85) * half
            self.freq[i] = rng.uniform(0.05, 0.25)   # Hz, slow & smooth
            self.phase[i] = rng.uniform(0, 2 * math.pi)

    def q(self, t):
        val = self.center + self.amp * np.sin(
            2 * math.pi * self.freq * t + self.phase)
        return np.clip(val, self.lo, self.hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    project_root = Path(__file__).resolve().parent.parent
    rng = np.random.default_rng(args.seed)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "collected_dataset" / f"roarm_{ts}"
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {output_dir}")

    # ---- Import robot to a cached USD, then reference it into the world ----
    robot_usd = import_robot_usd(args.urdf_path, args.robot_usd)

    world = World(stage_units_in_meters=1.0)
    stage = omni_usd.get_context().get_stage()

    robot_prim_path = "/World/RoArm"
    add_reference_to_stage(robot_usd, robot_prim_path)  # base fixed at origin
    deinstance_subtree(stage, robot_prim_path)          # make visual meshes render
    recolor_robot_dark_gray(stage, robot_prim_path)
    build_table(stage)
    setup_lights(stage)
    cube_paths = build_cube_pool(stage, args.max_cubes)

    # ---- Fixed camera ----
    cam_quat = look_at_quat(args.cam_pos, args.cam_target)
    camera = Camera(
        prim_path="/World/FixedCamera",
        position=np.array(args.cam_pos, float),
        orientation=cam_quat,
        frequency=args.fps,
        resolution=(args.width, args.height),
    )

    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()
    set_camera_fov(stage, "/World/FixedCamera", args.fov_h, args.width, args.height)
    camera.set_world_pose(position=np.array(args.cam_pos, float),
                          orientation=cam_quat)

    # ---- Articulation ----
    robot = SingleArticulation(prim_path=robot_prim_path, name="roarm")
    robot.initialize()
    dof_names = list(robot.dof_names)
    log(f"Robot DOF ({robot.num_dof}): {dof_names}")

    # Prime the render product so the RGB/depth annotators return real frames.
    for _ in range(5):
        world.step(render=True)

    intr = read_intrinsics(camera, args.width, args.height, args.fov_h)
    log(f"Camera pos={args.cam_pos} target={args.cam_target} "
        f"fov={args.fov_h} res={args.width}x{args.height}")
    log(f"Intrinsics: {intr}")

    intrinsics_meta = {}
    frame_count = 0
    name_index = 0
    dt = 1.0 / args.fps

    for ep in range(args.num_episodes):
        if not simulation_app.is_running():
            break
        log(f"\n=== Episode {ep + 1}/{args.num_episodes} ===")

        randomize_table_material(stage, rng, ep)
        k = randomize_cubes(stage, cube_paths, rng, ep)
        randomize_lighting(stage, rng)
        traj = ArmTrajectory(dof_names, rng)

        # Cubes are static visuals (already resting on the table); just bring the
        # arm to its trajectory start over a few warmup steps.
        q0 = traj.q(0.0)
        zero_v = np.zeros(robot.num_dof)
        for _ in range(8):
            robot.set_joint_positions(q0)
            robot.set_joint_velocities(zero_v)
            robot.apply_action(ArticulationAction(joint_positions=q0))
            world.step(render=True)
        log(f"  cubes active: {k}; arm at start pose")

        for f in range(args.frames_per_episode):
            t = f * dt
            q = traj.q(t)
            # Drive targets + kinematic set => correct pose whether or not
            # position drives were authored on import.
            robot.apply_action(ArticulationAction(joint_positions=q))
            robot.set_joint_positions(q)
            robot.set_joint_velocities(zero_v)
            world.step(render=True)

            rgba = camera.get_rgba()
            depth = camera.get_depth()
            if (rgba is None or depth is None
                    or getattr(rgba, "ndim", 0) < 3 or rgba.size == 0
                    or getattr(depth, "ndim", 0) < 2 or depth.size == 0):
                continue

            if rgba.dtype == np.uint8:
                rgb_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            else:
                rgb_bgr = cv2.cvtColor(
                    (np.clip(rgba[:, :, :3], 0, 1) * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR)

            depth_clean = np.where(np.isfinite(depth), depth, 0.0)
            depth_mm = np.clip(depth_clean * 1000.0, 0,
                               args.max_depth_mm).astype(np.uint16)

            stem = f"{name_index:06d}"
            cv2.imwrite(str(rgb_dir / f"{stem}.png"), rgb_bgr)
            cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_mm)
            intrinsics_meta[stem] = intr

            frame_count += 1
            name_index += 1
            if frame_count % 50 == 0:
                log(f"  {frame_count} frames (episode {ep + 1}, "
                    f"frame {f + 1}/{args.frames_per_episode})")

        name_index += args.scene_gap  # sequence boundary

    # ---- Save intrinsics ----
    with open(output_dir / "intrinsics.json", "w") as fjson:
        json.dump(intrinsics_meta, fjson)
    with open(output_dir / "intrinsics.txt", "w") as ftxt:
        ftxt.write("Color Camera Intrinsics (fixed):\n")
        ftxt.write(f"  Width: {intr['width']}\n  Height: {intr['height']}\n")
        ftxt.write(f"  fx: {intr['fx']:.4f}\n  fy: {intr['fy']:.4f}\n")
        ftxt.write(f"  cx: {intr['cx']:.4f}\n  cy: {intr['cy']:.4f}\n")
        ftxt.write("\nDepth Scale: 0.001\n")
        ftxt.write(f"Max depth (mm): {args.max_depth_mm}\n")
        ftxt.write(f"Camera position: {args.cam_pos}\n")
        ftxt.write(f"Camera target: {args.cam_target}\n")
        ftxt.write("\nSource: Isaac Sim (RoArm tabletop, fixed camera)\n")

    log(f"\nDone! {frame_count} frames across {args.num_episodes} episodes "
        f"saved to {output_dir}")
    log("Run split_dataset.py next to create train/val/test splits.")

    # simulation_app.close() hangs forever on Blackwell (sm_120 / RTX 5080);
    # outputs are already written, so hard-exit (same workaround as verify_isaac.py).
    sys.stderr.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    import traceback
    try:
        from pxr import Usd  # noqa: F401  (kept for clarity; Usd_iter uses children)
        main()
    except Exception as e:
        sys.stderr.write(f"\n\nFATAL ERROR: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        simulation_app.close()
