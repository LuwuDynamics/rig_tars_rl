"""Rendering of the fun-motion clips: GIF + MP4 per clip, a strip of key frames, and an overview sheet.

A clip is a dict: name, title (Chinese), subtitle (English), env, frames [(t, qpos, label, extra)], view {azim, elev, dist, lookat_z, follow}, stats.
`extra` of a frame may carry decorations in world coordinates: {'deco': [('capsule', p0, p1, radius, rgba), ('sphere', p, radius, rgba)]}.
The left arm is drawn blue and the right arm orange so that the two arms can be told apart (the real robot is black).
"""
import math
import os
import subprocess

import mujoco
import numpy as np
import PIL.Image as I
import PIL.ImageDraw as D
import PIL.ImageFont as F

FONTS = ['/System/Library/Fonts/Hiragino Sans GB.ttc', '/System/Library/Fonts/PingFang.ttc', '/Library/Fonts/Arial Unicode.ttf', '/System/Library/Fonts/Helvetica.ttc']
ARM_L = (0.25, 0.55, 0.95, 1.0)
ARM_R = (0.98, 0.55, 0.15, 1.0)
_FONT_CACHE = {}


def font(size):
    if size not in _FONT_CACHE:
        for p in FONTS:
            if os.path.exists(p):
                try:
                    _FONT_CACHE[size] = F.truetype(p, size)
                    break
                except OSError:
                    continue
        else:
            _FONT_CACHE[size] = F.load_default()
    return _FONT_CACHE[size]


class ClipRenderer:
    def __init__(self, model, w=480, h=360, recolor=True, bright=True):
        self.m = model
        self.d = mujoco.MjData(model)
        self.w, self.h = w, h
        if recolor:
            for body, rgba in (('arm_l', ARM_L), ('arm_r', ARM_R)):
                bid = model.body(body).id
                for g in range(model.ngeom):
                    if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                        model.geom_rgba[g] = rgba
        if bright:
            model.vis.headlight.ambient[:] = 0.45
            model.vis.headlight.diffuse[:] = 0.75
            model.vis.headlight.specular[:] = 0.1
        self.r = mujoco.Renderer(model, height=h, width=w)
        self._look = None

    def close(self):
        self.r.close()

    def image(self, t, qpos, view, yaw0, title='', subtitle='', label='', T=None, extra=None):
        m, d = self.m, self.d
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = view.get('azim', 140) + math.degrees(yaw0), view.get('elev', -12), view.get('dist', 0.5)
        tgt = np.array([d.qpos[0], d.qpos[1]]) if view.get('follow', True) else np.array(view.get('center', (0.0, 0.0)))
        if self._look is None or not view.get('follow', True):
            self._look = tgt.copy()
        else:
            k = view.get('smooth', 0.15)
            self._look = self._look + k * (tgt - self._look)
        cam.lookat[:] = [self._look[0], self._look[1], view.get('lookat_z', 0.065)]
        self.r.update_scene(d, camera=cam)
        scn = self.r.scene
        for item in (extra or {}).get('deco', []):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            if item[0] == 'capsule':
                _, p0, p1, rad, rgba = item
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, rad, np.asarray(p0, float), np.asarray(p1, float))
            elif item[0] == 'sphere':
                _, p, rad, rgba = item
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, rad, rad]), np.asarray(p, float), np.eye(3).flatten(), np.array(rgba, np.float32))
            scn.ngeom += 1
        im = I.fromarray(self.r.render())
        dr = D.Draw(im)
        if title:
            dr.text((10, 6), title, font=font(22), fill=(255, 255, 255), stroke_width=2, stroke_fill=(20, 20, 20))
        if subtitle:
            ly, line = 34, ''
            for wd in subtitle.split(' '):
                if dr.textlength(line + wd, font=font(14)) > self.w - 20 and line:
                    dr.text((10, ly), line, font=font(14), fill=(235, 235, 160), stroke_width=1, stroke_fill=(20, 20, 20))
                    ly += 17
                    line = ''
                line += wd + ' '
            dr.text((10, ly), line, font=font(14), fill=(235, 235, 160), stroke_width=1, stroke_fill=(20, 20, 20))
        if T:
            qw, qx, qy, qz = qpos[3:7]
            pitch = math.degrees(math.atan2(-2 * (qx * qz - qw * qy), 1 - 2 * (qx * qx + qy * qy)))
            hud = 'hip L %+4.0f° R %+4.0f°   lift %+3.0f mm   pitch %+3.0f°' % (math.degrees(qpos[8]), math.degrees(qpos[9]), qpos[7] * 1000, pitch)
            dr.text((10, self.h - 24), hud, font=font(13), fill=(255, 255, 255), stroke_width=1, stroke_fill=(20, 20, 20))
            if label:
                dr.text((10, self.h - 44), label, font=font(15), fill=(160, 255, 200), stroke_width=1, stroke_fill=(20, 20, 20))
            dr.text((self.w - 70, self.h - 24), 't=%4.1fs' % t, font=font(14), fill=(255, 255, 255), stroke_width=1, stroke_fill=(20, 20, 20))
            dr.rectangle((0, self.h - 5, int(self.w * min(1.0, t / T)), self.h), fill=(255, 190, 60))
        return im


def yaw_of(q0):
    return math.atan2(2 * (q0[3] * q0[6] + q0[4] * q0[5]), 1 - 2 * (q0[5] ** 2 + q0[6] ** 2))


def render_clip(clip, outdir, w=480, h=360, gif=True, mp4=True, gif_size=(400, 300), gif_fps=15, strip_times=None, strip_cols=4, strip_size=(300, 225)):
    """MP4 at 50 frames per simulated second (x speed), GIF at `gif_fps` (x speed); both play at clip['speed'] times real time"""
    os.makedirs(outdir, exist_ok=True)
    env, frames, view = clip['env'], clip['frames'], clip['view']
    yaw0 = yaw_of(frames[0][1])
    T = frames[-1][0]
    speed = clip.get('speed', 1.0)
    title, sub = clip.get('title', ''), clip.get('subtitle', '')
    base = os.path.join(outdir, clip['name'])
    out = {}
    cr = ClipRenderer(env.m, w, h)
    proc = None
    if mp4:
        proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (w, h), '-r', '50', '-i', '-',
                                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', base + '.mp4'], stdin=subprocess.PIPE)
    step_mp4 = speed                                      # 50 fps playback: one frame per 20 ms of video -> advance `speed` simulation steps
    i = 0.0
    k = 0
    while k < len(frames):
        t, q, lab, extra = frames[k]
        im = cr.image(t, q, view, yaw0, title, sub, lab, T, extra)
        if proc:
            proc.stdin.write(im.tobytes())
        i += 1.0
        k = int(round(i * step_mp4))
    if proc:
        proc.stdin.close()
        proc.wait()
        out['mp4'] = base + '.mp4'
        if gif:
            subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', base + '.mp4', '-vf',
                                   'fps=%d,scale=%d:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle' % (gif_fps, gif_size[0]),
                                   '-loop', '0', base + '.gif'])
            out['gif'] = base + '.gif'
    cr.close()
    if strip_times:
        out['strip'] = render_strip(clip, base + '_strip.png', strip_times, strip_cols, strip_size)
    return out


def key_times(clip, k=6, skip_end=0.4):
    """k frame times that cover the clip's variety: farthest-point sampling on (hips, lift, torso pose, position), the first frame always included"""
    frames = clip['frames']
    T = frames[-1][0]
    idx = [i for i, f in enumerate(frames) if f[0] <= T - skip_end] or list(range(len(frames)))
    feats = []
    for i in idx:
        q = frames[i][1]
        w_, x_, y_, z_ = q[3:7]
        up = np.array([2 * (x_ * z_ + w_ * y_), 2 * (y_ * z_ - w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)])        # torso z axis in the world
        fwd = np.array([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ + w_ * z_), 2 * (x_ * z_ - w_ * y_)])        # torso x axis in the world
        feats.append(np.concatenate([[q[8] / 1.2, q[9] / 1.2, q[7] / 0.014 * 0.15], up * 1.0, fwd * 0.9, [q[0] / 0.08, q[1] / 0.08]]))
    feats = np.array(feats)
    sel = [0]
    dmin = np.linalg.norm(feats - feats[0], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(dmin))
        sel.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(feats - feats[j], axis=1))
    return sorted(frames[idx[j]][0] for j in sel)


def render_strip(clip, path, times, cols=4, size=(300, 225)):
    env, frames, view = clip['env'], clip['frames'], clip['view']
    yaw0 = yaw_of(frames[0][1])
    cr = ClipRenderer(env.m, size[0], size[1])
    tl = [f[0] for f in frames]
    sel = []
    for ts in times:
        k = int(np.argmin(np.abs(np.array(tl) - ts)))
        t, q, lab, extra = frames[k]
        cr._look = None
        sel.append(cr.image(t, q, view, yaw0, '', '', lab, None, extra))
    cr.close()
    rows = (len(sel) + cols - 1) // cols
    sheet = I.new('RGB', (cols * size[0], rows * size[1]))
    for i, im in enumerate(sel):
        d = D.Draw(im)
        d.text((6, 4), '%.1fs' % times[i], font=font(14), fill=(255, 255, 255), stroke_width=2, stroke_fill=(20, 20, 20))
        sheet.paste(im, ((i % cols) * size[0], (i // cols) * size[1]))
    sheet.save(path)
    return path


def overview(clips, path, frames_per_clip=6, size=(240, 180), label_w=230, margin=4):
    """one row per clip: the title on the left, `frames_per_clip` key frames (farthest-point sampled over the clip)"""
    rows = len(clips)
    W = label_w + frames_per_clip * size[0] + margin * 2
    H = rows * size[1] + margin * 2
    sheet = I.new('RGB', (W, H), (28, 30, 36))
    dr = D.Draw(sheet)
    for r_, clip in enumerate(clips):
        env, frames, view = clip['env'], clip['frames'], clip['view']
        yaw0 = yaw_of(frames[0][1])
        cr = ClipRenderer(env.m, size[0], size[1])
        y = margin + r_ * size[1]
        dr.text((12, y + 10), '%d  %s' % (r_ + 1, clip.get('title', '')), font=font(26), fill=(255, 255, 255))
        sub = clip.get('subtitle', '')
        words, line, ly = sub.split(' '), '', y + 50
        for wd in words:
            if len(line) + len(wd) > 30:
                dr.text((12, ly), line, font=font(13), fill=(235, 235, 160)); ly += 17; line = ''
            line += wd + ' '
        dr.text((12, ly), line, font=font(13), fill=(235, 235, 160))
        for c_, ts in enumerate(key_times(clip, frames_per_clip)):
            k = int(np.argmin([abs(f[0] - ts) for f in frames]))
            t, q, lab, extra = frames[k]
            cr._look = None
            im = cr.image(t, q, view, yaw0, '', '', '', None, extra)
            sheet.paste(im, (margin + label_w + c_ * size[0], y))
        cr.close()
    sheet.save(path)
    return path
