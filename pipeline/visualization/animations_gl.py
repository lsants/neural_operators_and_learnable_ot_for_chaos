from manimlib import *
import numpy as np
import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from models.operator.mlp_timestepper import TimeStepperMLP
from models.summary.mlp_summary import MLPSummaryStats
from pipeline.testing.eval_rollout import eval_rollout

# Example
dataset_path = '/Users/ls/workspace/neural_operators_and_learnable_ot_for_chaos/data/lorenz63/3e39e796/test_data.npz'
checkpoint_path = '/Users/ls/workspace/neural_operators_and_learnable_ot_for_chaos/outputs/lorenz63_20251029_211138_mlp_noisy_6a42d5d5/6a42d5d5/checkpoints/_epoch_100.pt'

class Lorenz63(InteractiveScene):
    def construct(self):
        true_data = load_dataset(dataset_path)
        true_traj, true_param = get_trajectory_and_param(true_data, 0)

        emulator, summary = get_emulator_and_summary(checkpoint_path)
        
        pred_data = eval_rollout(emulator, torch.tensor(true_traj[None, :, :], dtype=torch.float32), torch.tensor(true_param[None, :], dtype=torch.float32) ).squeeze(0).detach().numpy()
        
        pred_traj = pred_data
        summary_traj = summary(torch.tensor(true_traj[None, :, :], dtype=torch.float32)).squeeze(0).detach().numpy()
        

        points_true = true_traj
        points_pred = pred_traj
        points_summary = summary_traj

        axes = ThreeDAxes(
            x_range=(-50, 50, 5),
            y_range=(-50, 50, 5),
            z_range=(-0, 50, 5),
            width=16,
            height=16,
            depth=8,
        )
        axes.set_width(FRAME_WIDTH)
        axes.center()
        self.frame.reorient(40, 90, 1, IN, 10)
        self.add(axes)

        trajs = [points_true, points_pred] # , points_summary]
        colors = [interpolate_color(BLUE, YELLOW, i / len(trajs)) for i in range(len(trajs))]

        curves = VGroup()

        for var, color in zip(trajs, colors):
            curve = VMobject().set_points_as_corners(axes.c2p(*var.T))
            curve.set_stroke(color, 2)
            curves.add(curve)

        dots = Group(GlowDot(ORIGIN, color=color, radius=0.25) for color in colors)

        def update_dots(dots, curves=curves):
            for dot, curve in zip(dots, curves):
                dot.move_to(curve.get_end())
        dots.add_updater(update_dots)

        tail_true = TracingTail(dots[0], stroke_color=colors[0])
        tail_pred = TracingTail(dots[1], stroke_color=colors[1])
        # tail_summary = TracingTail(dots[2], stroke_color=colors[2])


        self.add(dots)
        self.add(tail_true)
        self.add(tail_pred)
        # self.add(tail_summary)
        
        curves.set_opacity(0)

        self.play(*(ShowCreation(curve, rate_func=linear)
            for curve in curves),
            # self.frame.animate.reorient(70, 90, 1, IN, 100),
            # FadeOut(curves),
            run_time=100
        )
        self.wait()


def load_dataset(path: str):
    data = np.load(path)
    return data

def get_trajectory_and_param(data, index: int):
    key_traj = f"traj_{str(index).zfill(6)}"
    key_param = f"params_{str(index).zfill(6)}"
    return data[key_traj], data[key_param]

def get_emulator_and_summary(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path)
    emulator = TimeStepperMLP(**checkpoint['emulator_config']).to(dtype=torch.float32)
    summary = MLPSummaryStats(**checkpoint['summary_config']).to(dtype=torch.float32)
    summary.load_state_dict(checkpoint['summary_state_dict'])
    emulator.load_state_dict(checkpoint['model_state_dict'])
    return emulator, summary

if __name__ == '__main__':
    data = load_dataset(dataset_path)
    true_data = load_dataset(dataset_path)
    true_traj, true_param = get_trajectory_and_param(true_data, 0)

    emulator, summary = get_emulator_and_summary(checkpoint_path)
    
    pred_data = eval_rollout(emulator, torch.tensor(true_traj[None, :, :], dtype=torch.float32), torch.tensor(true_param[None, :], dtype=torch.float32) ).squeeze(0).detach().numpy()
    
    pred_traj = pred_data
    summary_traj = summary(torch.tensor(true_traj[None, :, :], dtype=torch.float32)).squeeze(0).detach().numpy()
    print(true_traj[0], pred_traj[0], summary_traj[0])
