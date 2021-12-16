import argparse
import os


def str2bool(x):
    try:
        return bool(int(x))
    except:
        pass
    s = str(x).lower()
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        raise Excetion(f"Invalid value {x}")


def whichgame_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_workers", type=int, default=os.environ.get("NUM_WORKERS", 24)
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("DATA_DIR", "/ssd1/whichgame-v3-dataset"),
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=os.environ.get("CKPT_DIR", "/bigdata0/checkpoints"),
    )
    parser.add_argument(
        "--ingame_subdir",
        type=str,
        default=os.environ.get("INGAME_SUBDIR", "ingame"),
    )
    parser.add_argument(
        "--outgame_subdir",
        type=str,
        default=os.environ.get("OUTGAME_SUBDIR", "outgame"),
    )
    parser.add_argument(
        "--outgame_cats_subdir",
        type=str,
        default=os.environ.get("OUTGAME_CATS_SUBDIR", "outgame-categories"),
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=os.environ.get(
            "LABELS_DIR",
            f"{os.environ['HOME']}/labeling-efforts/whichgame-concat",
        ),
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=os.environ.get("COMET_PROJ_NAME", "whichgame-tests"),
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=os.environ.get("COMET_EXP_NAME", "mobilenet-outofline"),
    )
    """parser.add_argument(
        "--valid_twids_path",
        type=str,
        default="../datasets/valid_twids/2021.06.08-15.12_valid-propn=0.2_test-propn=0.1.txt",
    )"""
    parser.add_argument(
        "--valid_twids_path",
        type=str,
        default=os.environ.get(
            "VALID_TWIDS_PATH",
            f"{os.environ['HOME']}/whichgame_project/whichgame-ai-liticia/datasets/valid_twids/2021.06.08-15.12_valid-propn=0.2_test-propn=0.1.txt",
        ),
    )
    parser.add_argument(
        "--test_twids_path",
        type=str,
        default=os.environ.get(
            "TEST_TWIDS_PATH",
            f"{os.environ['HOME']}/whichgame_project/whichgame-ai-liticia/datasets/valid_twids/2021.06.08-15.12_valid-propn=0.2_test-propn=0.1_test.txt",
        ),
    )
    """
    parser.add_argument(
        "--test_twids_path",
        type=str,
        default="../datasets/valid_twids/2021.06.08-15.12_valid-propn=0.2_test-propn=0.1_test.txt",
    )"""
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=360)
    parser.add_argument(
        "--backbone_name", type=str, default="mobilenet_v2",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--pretrained", type=str2bool, default=True)
    parser.add_argument("--dropout_fts", type=float, default=0.1)

    parser.add_argument("--ckpt_file", type=str, default='epoch-epoch=00049.ckpt')
    parser.add_argument("--ckpt_file_ac", type=str, default='mobilenet-outofline_cf1cb73_calibrated.pth')

    parser.add_argument("--num_ingame_classes", type=int, default=80)
    parser.add_argument("--num_outgame_classes", type=int, default=2)
    parser.add_argument("--ingame_classes", type=list, default=[])
    parser.add_argument("--outgame_classes", type=list, default=[])

    parser.add_argument("--model_name", type=str, default='net')
    parser.add_argument("--calibrator_name", type=str, default='calibrateWithFocallLoss')
    parser.add_argument("--gamma", type=int, default=1)
    return parser
