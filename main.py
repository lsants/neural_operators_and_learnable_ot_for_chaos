import argparse
import logging
from pathlib import Path
from pipeline.training import train, get_train_configs
from pipeline.testing import evaluate, get_eval_configs

logger = logging.getLogger(__file__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end emulator")
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--eval-config",   type=Path, required=False)   # optional
    parser.add_argument("--operator-config", type=Path, required=True)
    parser.add_argument("--loss-config",      type=Path, required=True)
    parser.add_argument("--run-eval", action="store_true", help="Run evaluation after training")
    return parser.parse_args()
 
def main():
    args = _parse_args()

    # ----------- Training ---------------------
    train_config = get_train_configs(
        config_paths=[
            args.train_config, 
            args.operator_config, 
            args.loss_config, 
        ]
    )
    # eval_config = get_eval_configs(args.eval_config)

    train(train_config, use_wandb=True)

    # --------- Optional Evaluation --------------
    if args.run_eval:
            if args.eval_config is None:
                raise ValueError("--eval-config is required when --run-eval is used")
            # eval_cfg = get_eval_configs(args.eval_config)
            # evaluate(eval_cfg)

if __name__ == '__main__':
    main()