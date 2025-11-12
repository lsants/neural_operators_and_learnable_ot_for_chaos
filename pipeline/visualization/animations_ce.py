import numpy as np
from manim import *

class Lorenz63(ThreeDScene):
    def __init__(self, true_traj, pred_traj, summary_traj):
        super().__init__()
        self.true_traj = true_traj
        self.pred_traj = pred_traj
        self.summary_traj = summary_traj

    def construct(self):
        # --- Load data ---
        points_true = self.true_traj
        points_pred = self.pred_traj
        points_summary = self.summary_traj

        # --- Axes setup ---
        axes = ThreeDAxes(
            x_range=(-60, 60, 6),
            y_range=(-60, 60, 6),
            z_range=(0, 50, 5),
            x_length=10,
            y_length=10,
            z_length=5,
        )

        self.set_camera_orientation(
            phi=70 * DEGREES, 
            theta=135 * DEGREES, 
            gamma=0 * DEGREES, 
            zoom=0.9,
            frame_center=OUT
        )
        self.add(axes)

        trajs = [points_true, points_pred, points_summary]
        colors = [YELLOW, PINK, BLUE]

        # === Fade controls (feel like ManimGL set_opacity) ===
        fade_window  = 0.1     # arc-length fraction that remains visible at ~full opacity
        fade_gamma   = 1.0      # >1 = sharper fade, <1 = softer
        max_opacity  = 1.0
        min_opacity  = 0.0
        width_near   = 3.0
        width_far    = 0.8

        # Optional resampling density along arc-length (helps smooth opacity)
        target_vertices = 600  # per trajectory; tune as needed

        def arclen_param(P: np.ndarray):
            """Return normalized cumulative arc-length s ∈ [0,1] for an Nx3 polyline."""
            seg = np.linalg.norm(P[1:] - P[:-1], axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            total = max(cum[-1], 1e-9)
            return cum / total, total

        def resample_by_arclen(P: np.ndarray, n: int):
            """Resample polyline P (Nx3) to n points uniformly in arc-length."""
            s, _ = arclen_param(P)
            u = np.linspace(0.0, 1.0, n)
            out = np.empty((n, 3))
            for j, uj in enumerate(u):
                i = int(np.searchsorted(s, uj, side="right") - 1)
                i = np.clip(i, 0, len(P) - 2)
                denom = (s[i + 1] - s[i]) if s[i + 1] > s[i] else 1.0
                t = (uj - s[i]) / denom
                out[j] = (1 - t) * P[i] + t * P[i + 1]
            return out, u

        def build_full_fade_tail(sampled_pts: np.ndarray, sgrid: np.ndarray, alpha: ValueTracker, color=WHITE):
            """
            Build a VGroup of segments covering the whole trajectory every frame.
            - Segments *ahead* of the dot have opacity 0 (invisible).
            - Segments *behind* fade with age relative to 'fade_window'.
            - The final partial segment ends exactly at the dot (no desync).
            """
            def profile(age: float) -> float:
                if age <= 0.0:
                    return 0.0
                x = min(age / fade_window, 1.0)
                return (1.0 - x) ** fade_gamma

            def redraw():
                u_end = float(alpha.get_value())
                pts = sampled_pts
                u  = sgrid

                group = VGroup()

                for k in range(len(pts) - 1):
                    u_mid = 0.5 * (u[k] + u[k + 1])
                    age   = u_end - u_mid
                    w = width_near * profile(age) + width_far * (1.0 - profile(age))
                    op = max_opacity * profile(age) + min_opacity * (1.0 - profile(age))
                    if op <= 0.0:
                        continue

                    if u[k] <= u_end <= u[k + 1]:
                        denom = (u[k + 1] - u[k]) if u[k + 1] > u[k] else 1.0
                        t = (u_end - u[k]) / denom
                        p_end = (1 - t) * pts[k] + t * pts[k + 1]
                        seg = Line(pts[k], p_end).set_stroke(color=color, width=w, opacity=op)
                        group.add(seg)
                    elif u[k + 1] <= u_end:
                        seg = Line(pts[k], pts[k + 1]).set_stroke(color=color, width=w, opacity=op)
                        group.add(seg)
                    else:
                        # segment ahead of dot -> opacity already ~0 via profile(age<=0)
                        pass

                return group

            return always_redraw(redraw)

        # === Build everything (hidden precomputation + visible dot/tail) ===
        alphas: list[ValueTracker] = []
        dots   = VGroup()
        tails  = VGroup()

        for arr, color in zip(trajs, colors):
            # map to scene coords and resample uniformly by arc-length for stable fading
            P_scene = np.asarray([axes.c2p(*p) for p in arr], dtype=float)
            P_uni, u_uni = resample_by_arclen(P_scene, target_vertices)

            alpha = ValueTracker(0.0)
            alphas.append(alpha)

            # dot follows arc-length proportion exactly
            def dot_pos(a=alpha, pts=P_uni, uu=u_uni):
                u_end = float(a.get_value())
                i = int(np.searchsorted(uu, u_end, side="right") - 1)
                i = np.clip(i, 0, len(pts) - 2)
                denom = (uu[i + 1] - uu[i]) if uu[i + 1] > uu[i] else 1.0
                t = (u_end - uu[i]) / denom
                return (1 - t) * pts[i] + t * pts[i + 1]
            
            def dot_pos(a=alpha, pts=P_uni, uu=u_uni):
                u_end = float(a.get_value())
                i = int(np.searchsorted(uu, u_end, side="right") - 1)
                i = np.clip(i, 0, len(pts) - 2)
                denom = (uu[i + 1] - uu[i]) if uu[i + 1] > uu[i] else 1.0
                t = (u_end - uu[i]) / denom
                return (1 - t) * pts[i] + t * pts[i + 1]

            if color == BLUE:
                dot = Dot3D(color=color, radius=0.03).move_to(P_uni[0])
            else:
                dot = Dot3D(color=color, radius=0.06).move_to(P_uni[0])
            dot.add_updater(lambda m, f=dot_pos: m.move_to(f()))
            dots.add(dot)

            # full-trajectory tail with time-based fading (no cropping)
            tail = build_full_fade_tail(P_uni, u_uni, alpha, color=color)
            tails.add(tail)

        # draw order: tails first (under), dots on top (leaders)
        self.add(tails, dots)

        # === Single play drives everything (perfect sync through shared arclength) ===
        self.play(
            *[a.animate.set_value(1.0) for a in alphas],
            run_time=15,
            rate_func=linear,
        )
        self.wait()


    def load_dataset(self):
        data = np.load(self.dataset_path)
        return data

    def get_trajectory_and_param(self, data, index: int):
        key_traj = f"traj_{str(index).zfill(6)}"
        key_param = f"params_{str(index).zfill(6)}"
        return data[key_traj], data[key_param]

