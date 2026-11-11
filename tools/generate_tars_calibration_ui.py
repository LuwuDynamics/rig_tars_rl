#!/usr/bin/env python3
"""Render the TARS reset countdown and calibration guide for its 320x240 LCD.

Run with the tars_rl Python environment (MuJoCo, NumPy, Pillow). The output
goes into the board's Flash assets. MuJoCo is only used by this offline tool.
"""

import argparse
from functools import partial
import math
from pathlib import Path
import struct

import mujoco
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont


WIDTH, HEIGHT = 320, 240
COUNTDOWN_SECONDS = 10
COUNTDOWN_FRAMES = COUNTDOWN_SECONDS + 1
FRAME_MS = 125
STEP_FRAMES = 24
BG = "#0a1017"
TEXT = "#eef6ff"
SECONDARY = "#9cadc0"
MINT = "#87f5cc"
AMBER = "#ffc367"
TRACK = "#273644"
RING_COLORS = ("#88f5c9", "#88f5c9", "#9cf0b9", "#b2ec9f", "#d8e68b",
               "#f5d876", "#ffbf68", "#ffaa61", "#ff9563", "#ff8066")

COPY = {
    "zh": (
        ("长按重置", "松手取消", "按满 10 秒清除 Wi-Fi 和标定", "正在清除 Wi-Fi 和标定"),
        "标定指引",
        ("将机身直立放稳", "摆正左右两条腿", "轻点顶部中间的触摸键"),
        ("放在平整桌面，保持机身竖直", "两条腿对齐，底边贴平桌面", "保持机器不动，等待标定完成"),
    ),
    "en": (
        ("HOLD TO RESET", "Release to cancel", "Hold 10s to clear Wi-Fi and calibration", "Clearing Wi-Fi and calibration"),
        "CALIBRATION GUIDE",
        ("Stand TARS upright", "Align both legs", "Tap the top center touch key"),
        ("Use a flat surface and keep the body vertical", "Keep both feet flat and aligned", "Hold TARS still while it calibrates"),
    ),
}


def countdown(font, labels):
    title, cancel, detail, clearing = labels
    scale = 2
    title_font = font(17 * scale)
    digit_font = font(68 * scale)
    hint_font = font(14 * scale)
    for seconds in range(COUNTDOWN_SECONDS, -1, -1):
        frame = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), BG)
        draw = ImageDraw.Draw(frame)
        draw.text((320, 37), title, font=title_font, fill=TEXT, anchor="mm")
        for i, color in enumerate(RING_COLORS):
            draw.arc((174, 72, 466, 364), -87 + i * 36, -57 + i * 36,
                     fill=color if i < COUNTDOWN_SECONDS - seconds else TRACK, width=18)
        draw.text((320, 218), str(seconds), font=digit_font, fill=TEXT, anchor="mm")
        if seconds:
            draw.text((320, 400), cancel, font=hint_font, fill=MINT, anchor="mm")
            draw.text((320, 438), detail, font=hint_font, fill=SECONDARY, anchor="mm")
        else:
            draw.text((320, 421), clearing, font=hint_font, fill=AMBER, anchor="mm")
        yield frame.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def add_capsule(scene, start, end, radius, color):
    geom = scene.geoms[scene.ngeom]
    kind = mujoco.mjtGeom.mjGEOM_CAPSULE
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
    mujoco.mjv_connector(geom, kind, radius, np.array(start), np.array(end))
    scene.ngeom += 1


def model_frames(model_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.vis.headlight.ambient[:] = 0.55
    model.vis.headlight.diffuse[:] = 0.8
    model.geom("floor").matid[0] = -1
    model.geom("floor").group[0] = 1
    model.geom("floor").size[:2] = [0.16, 0.14]
    model.geom("floor").rgba[:] = [0.055, 0.08, 0.11, 1]
    shell_meshes = {model.mesh(name).id for name in ("torso_body", "torso_front", "arm_l", "arm_r")}
    for i, kind in enumerate(model.geom_type):
        if kind == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[i] in shell_meshes:
            model.geom_rgba[i] = [0.43, 0.48, 0.55, 1]
    data = mujoco.MjData(model)
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0
    opt.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, 0.09]
    camera.azimuth, camera.elevation, camera.distance = 155, -16, 0.30
    with mujoco.Renderer(model, height=296, width=640) as renderer:
        for step in range(3):
            for i in range(STEP_FRAMES):
                fraction = min(i / 10, 1.0)
                eased = fraction * fraction * (3 - 2 * fraction)
                data.qpos[:] = model.qpos0
                if step == 0:
                    angle = math.radians(19 * (1 - eased))
                    data.qpos[3:7] = [math.cos(angle / 2), 0, math.sin(angle / 2), 0]
                if step == 1:
                    data.joint("hip_l").qpos[0] = 0.39 * (1 - eased)
                    data.joint("hip_r").qpos[0] = -0.35 * (1 - eased)
                mujoco.mj_forward(model, data)
                data.qpos[2] -= min(data.site("foot_l").xpos[2], data.site("foot_r").xpos[2])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=opt)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False

                if step == 0:
                    for z in np.linspace(0.009, 0.151, 10):
                        add_capsule(renderer.scene, (0, 0.092, z), (0, 0.092, z + 0.006),
                                    0.0014, [0.23, 0.83, 0.65, 1])
                elif step == 1:
                    for leg in ("l", "r"):
                        foot = data.site(f"foot_{leg}").xpos.copy()
                        start = foot + np.array([-0.029, 0, 0.004])
                        end = foot + np.array([0.017, 0, 0.004])
                        add_capsule(renderer.scene, start, end, 0.002, [0.23, 0.83, 0.65, 1])
                else:
                    top = data.xpos[model.body("torso").id, 2] + 0.073
                    color = [1.0, 0.65, 0.18, 1]
                    bounce = 0.007 * (1 + math.cos(i * math.pi / 5))
                    tip = top + 0.012 + bounce
                    add_capsule(renderer.scene, (0, 0, tip + 0.023), (0, 0, tip), 0.0022, color)
                    for side in (-1, 1):
                        add_capsule(renderer.scene, (0, 0, tip), (0, side * 0.009, tip + 0.009),
                                    0.0022, color)
                    add_capsule(renderer.scene, (0, -0.004, top + 0.002),
                                (0, 0.004, top + 0.002), 0.0025, color)
                yield Image.fromarray(renderer.render()).resize((WIDTH, 148), Image.Resampling.LANCZOS)


def guide(frames, font, labels):
    _, title, steps, details = labels
    title_font = font(16)
    step_font = font(20)
    detail_font = font(14)
    for index, robot in enumerate(frames):
        step = index // STEP_FRAMES
        frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
        frame.paste(robot, (0, 34))
        draw = ImageDraw.Draw(frame)
        draw.text((14, 17), title, font=title_font, fill=TEXT, anchor="lm")
        draw.text((305, 17), f"{step + 1} / 3", font=title_font, fill=MINT, anchor="rm")
        draw.text((160, 201), steps[step], font=step_font, fill=TEXT, anchor="mm")
        draw.text((160, 225), details[step], font=detail_font, fill=SECONDARY, anchor="mm")
        yield frame


def pack_indices(indices):
    """Packets of 1..128 indices. High bit repeats the next index."""
    result = bytearray()
    literal = bytearray()
    pos = 0
    while pos < len(indices):
        count = 1
        while count < 128 and pos + count < len(indices) and indices[pos] == indices[pos + count]:
            count += 1
        if count >= 3 or len(literal) == 128:
            if literal:
                result.append(len(literal) - 1)
                result.extend(literal)
                literal.clear()
        if count >= 3:
            result.extend((0x80 | (count - 1), indices[pos]))
            pos += count
        else:
            literal.append(indices[pos])
            pos += 1
    if literal:
        result.append(len(literal) - 1)
        result.extend(literal)
    return bytes(result)


def write_asset(path, frames):
    # One palette for the entire loop. Sample each frame so text and markers
    # from all steps survive quantization. Avoid dithering to keep runs long.
    samples = Image.new("RGB", (160, 120 * len(frames)))
    for i, frame in enumerate(frames):
        samples.paste(frame.resize((160, 120)), (0, i * 120))
    fixed = [BG, TEXT, SECONDARY, MINT, AMBER, TRACK, *dict.fromkeys(RING_COLORS)]
    remaining = 256 - len(fixed)
    colors = samples.quantize(colors=remaining).getpalette()[:3 * remaining]
    palette = Image.new("P", (1, 1))
    palette.putpalette(colors + [channel for color in fixed for channel in ImageColor.getrgb(color)])
    colors = np.array(palette.getpalette(), dtype=np.uint16).reshape(256, 3)
    rgb565 = ((colors[:, 0] >> 3) << 11) | ((colors[:, 1] >> 2) << 5) | (colors[:, 2] >> 3)
    header = struct.pack("<4s4H", b"TCG1", WIDTH, HEIGHT, len(frames) - COUNTDOWN_FRAMES, FRAME_MS)
    table = bytearray()
    payload = bytearray()
    offsets = {}
    quantized = []
    for frame in frames:
        indexed = frame.quantize(palette=palette, dither=Image.Dither.NONE)
        raw = indexed.tobytes()
        if raw not in offsets:
            packed = pack_indices(raw)
            offsets[raw] = len(payload), len(packed)
            payload.extend(packed)
        table.extend(struct.pack("<II", *offsets[raw]))
        quantized.append(indexed)
    path.write_bytes(header + rgb565.astype(">u2").tobytes() + table + payload)
    print(f"{path.name}: {path.stat().st_size:,} bytes")
    return quantized


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=repo.parent / "tars_rl/model/tars.xml")
    parser.add_argument("--font-zh", required=True)
    parser.add_argument("--font-zh-index", type=int, default=0,
                        help="Face index for Simplified Chinese in a TrueType collection")
    parser.add_argument("--font-en", default="DejaVuSans.ttf")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    robot = list(model_frames(args.model))
    for locale, font_path, index in (("zh", args.font_zh, args.font_zh_index), ("en", args.font_en, 0)):
        font = partial(ImageFont.truetype, font_path, index=index)
        frames = list(countdown(font, COPY[locale][0])) + list(guide(robot, font, COPY[locale]))
        path = repo / "main/boards/tars/320_240" / f"tars_calib_ui_{locale}.tcg"
        quantized = write_asset(path, frames)
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            for name, part, duration in (
                ("countdown", quantized[:COUNTDOWN_FRAMES], 1000),
                ("guide", quantized[COUNTDOWN_FRAMES:], FRAME_MS),
            ):
                part[0].save(args.preview_dir / f"tars-{name}-{locale}.gif",
                             save_all=True, append_images=part[1:], duration=duration, loop=0)


if __name__ == "__main__":
    main()
