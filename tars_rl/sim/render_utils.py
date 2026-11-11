import numpy as np, mujoco
import PIL.Image as I, PIL.ImageDraw as D

def render_strip(m, frames, fname, cols=4, w=320, h=240, azim=90, elev=-8, dist=0.42, lookat_follow=True, labels=None):
    """frames: list of (qpos array) -> grid image. camera looks from +Y side (side view of sagittal plane)."""
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=h, width=w)
    imgs = []
    for i, q in enumerate(frames):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = azim, elev, dist
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.06] if lookat_follow else [0, 0, 0.06]
        r.update_scene(d, camera=cam)
        im = I.fromarray(r.render())
        if labels:
            dr = D.Draw(im); dr.text((6, 4), labels[i], fill=(255, 255, 0))
        imgs.append(im)
    rows = (len(imgs) + cols - 1) // cols
    sheet = I.new('RGB', (cols * w, rows * h))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(fname)
    r.close()
