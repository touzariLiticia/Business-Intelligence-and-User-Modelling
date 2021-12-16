import comet_ml 
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from glob import glob
import os
from onnx import onnx_pb
from coremltools.converters.onnx import convert
from sklearn.metrics import roc_auc_score
from pytorch_lightning import Trainer
import torch
from torch.nn import functional as F
import torch.utils.data as tud
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.core.lightning import LightningModule
from tqdm import tqdm
import json
import argparse
import pylab as plt
import numpy as np
from calibrate_metrics import *
from calibration_models import *
import sys

root_path = os.path.abspath(os.path.realpath(__file__)+'/../../..')
sys.path.append(root_path)
from train.args import whichgame_args
from train.net.network import Net
from data_util.dataset import DatasetGetter
from data_util.data_utils import populate_classes_lists

def get_commit_hash():
    with open("../commithash", "r") as f:
        commithash = f.read()
        if commithash.endswith("\n"):
            commithash = commithash[:-1]
    return commithash


def inference_dataset(dataset, model, args, every=1, ret_xs=False, ret_just_one_x=False):
    if every > 1:
        dataset = tud.Subset(0, len(dataset), every)
    kwargs = dict(
        pin_memory=args.device != "cpu", num_workers=args.num_workers
    )
    loader = tud.DataLoader(dataset, batch_size=args.batch_size, **kwargs)

    if ret_just_one_x:
        ret_xs = True
    model.eval()
    (
        xs,
        ys_ingame,
        ys_whichgame,
        ys_subclass_if_outgame,
        ps_ingame,
        ps_whichgame,
        ps_subclass_if_outgame,
        fts,
    ) = [[] for _ in range(8)]
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["images"]
            y_ingame = batch["y_ingame"]
            y_whichgame = batch["y_whichgame"]
            y_subclass_if_outgame = batch["y_subclass_if_outgame"]
            (
                x_ingame,
                x_game,
                x_outgame_subclasses,
                ftrs_for_distill,
                x_ftrs,
            ) = model(x.to(args.device))
            r = x_ftrs
            p_ingame = x_ingame.softmax(1)
            p_whichgame = x_game.softmax(1)
            p_subclass_if_outgame = x_outgame_subclasses.softmax(1)
            if ret_xs:
                if not ret_just_one_x or not len(xs):
                    xs.append(x.cpu())
            fts.append(r)
            ys_ingame.append(y_ingame)
            ys_whichgame.append(y_whichgame)
            ys_subclass_if_outgame.append(y_subclass_if_outgame)
            ps_ingame.append(p_ingame)
            ps_whichgame.append(p_whichgame)
            ps_subclass_if_outgame.append(p_subclass_if_outgame)
    (
        ys_ingame,
        ys_whichgame,
        ys_subclass_if_outgame,
        ps_ingame,
        ps_whichgame,
        ps_subclass_if_outgame,
        fts,
    ) = [
        torch.cat(_, 0)
        for _ in [
            ys_ingame,
            ys_whichgame,
            ys_subclass_if_outgame,
            ps_ingame,
            ps_whichgame,
            ps_subclass_if_outgame,
            fts,
        ]
    ]
    if ret_xs:
        xs = torch.cat(xs, 0)
        return (
            xs,
            ys_ingame,
            ys_whichgame,
            ys_subclass_if_outgame,
            ps_ingame,
            ps_whichgame,
            ps_subclass_if_outgame,
            fts,
        )
    return (
        ys_ingame,
        ys_whichgame,
        ys_subclass_if_outgame,
        ps_ingame,
        ps_whichgame,
        ps_subclass_if_outgame,
        fts,
    )

class SoftmaxOutputPClass1(torch.nn.Module):
    def __init__(self):
        super(SoftmaxOutputPClass1, self).__init__()

    def forward(self, x):
        return torch.softmax(x, 1)

class WrapIt255(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        inv = torch.FloatTensor([1 / 0.229, 1 / 0.224, 1 / 0.225]) / 255.0
        mean = torch.FloatTensor([0.485, 0.456, 0.406]) * 255.0
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.conv.weight.data = torch.diag(inv).view(3, 3, 1, 1)
        self.conv.bias.data = -mean * inv
        self.m = m
        self.sm = SoftmaxOutputPClass1()

    def forward(self, x):
        x = self.conv(x)
        ig, which, og_subcats = self.m(x)[:3]
        return self.sm(ig), self.sm(which), self.sm(og_subcats)
class WrapIt(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.sm = SoftmaxOutputPClass1()

    def forward(self, x):
        ig, which, og_subcats = self.m(x)[:3]
        return self.sm(ig), self.sm(which), self.sm(og_subcats)

calibration_models = {
    'calibrateFC': CalibrateFC,
    'calibrateFCWhichGame': CalibrateFCWhichGame,
    'calibrateWithTemperature': CalibrateWithTemperature,
    'calibrateWithFocallLoss': CalibrateWithFocallLoss,
}

def train_calibrator(init_model, x_train, y_train, x_test, y_test, nb_classes, args, lr,
        weight_decay, type=''):
    
    calibrator = calibration_models[args.calibrator_name](
        x_train, 
        y_train, 
        x_test, 
        y_test,
        nb_classes,
        lr=lr,
        weight_decay=weight_decay,
        init_model=init_model,
    )
    if args.calibrator_name=="calibrateWithTemperature":
        print("Temperature scalling calibrator_"+type)
        calibrator = set_temperature(\
                        calibrator.to(args.device),\
                        calibrator.val_dataloader(),\
                        args.device)
    else:
        experiment = CometLogger(
            api_key="rEy6sPPJyZ6EATCASeGFbeNB8",
            project_name="whichgame-tests",
            experiment_name=str(args.calibrator_name)+type)

        trainer = Trainer(
            max_epochs=15, gpus=1, num_nodes=1, num_sanity_val_steps=-1, logger = experiment
        )
        print("Fitting calibrator_"+type)
        trainer.fit(calibrator)
    return calibrator

def train(args):

    # Load args and classes
    args.device = "cuda" if args.n_gpus > 0 else "cpu"
    args.ckpts_out_dir = os.path.join(
        args.ckpt_dir, args.project_name, args.experiment_name
    )
    (
        ingame_classes,
        outgame_classes,
        ingame_class_to_idx,
        outgame_class_to_idx,
    ) = populate_classes_lists(
        args.data_dir, args.ingame_subdir, args.outgame_cats_subdir
    )
    args.num_ingame_classes = len(ingame_classes)
    args.num_outgame_classes = len(outgame_classes)
    args.ingame_classes = ingame_classes
    args.outgame_classes = outgame_classes

    # Load up the "args.ckpt_file" checkpoint from the checkpoints dir
    list_of_files = glob(os.path.join(args.ckpts_out_dir, "*.ckpt"))
    assert len(list_of_files), f"No checkpoints found in {args.ckpts_out_dir}"

    ckpt_path = os.path.join(args.ckpts_out_dir, args.ckpt_file)
    print(f"Will load checkpoint from {ckpt_path}")

    model = Net(
        backbone_name=args.backbone_name,
        num_classes=len(ingame_classes),
        num_outgame_subclasses=len(outgame_classes),
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location="cpu")["state_dict"]
    )
    model = model.eval().to(args.device)
    print(f"Model state succesfully loaded from {ckpt_path}")

    ########################
    #    DATA
    ########################

    # Create a DatasetGetter: to get ingame or outgame datasets from "data_dir" 
    dataset_getter = DatasetGetter(
        args.data_dir,
        ingame_class_to_idx,
        args.labels_dir,
        args.ingame_subdir,
        args.outgame_subdir,
        args.outgame_cats_subdir,
        valid_test_twids_paths=(args.valid_twids_path, args.test_twids_path),
        outgame_class_to_idx=outgame_class_to_idx,
    )

    unlabeled_datasets = {}
    _ = dataset_getter.get_unlabeled_ingame_dataset("valid")
    # Reduce the size of the unlabeled dataset to make validation epoch faster
    unlabeled_valid_dataset_reduce_factor = 100
    ingame_unlabeled_reduced_dataset = tud.Subset(
        _, np.arange(0, len(_), unlabeled_valid_dataset_reduce_factor)
    )
    print(
        f"len(ingame_unlabeled_reduced_dataset) = {len(ingame_unlabeled_reduced_dataset)}"
    )

    (
        ys_ingame_u,
        ys_whichgame_u,
        ys_subclass_if_outgame_u,
        ps_ingame_u,
        ps_whichgame_u,
        ps_subclass_if_outgame_u,
        fts_u,
    ) = inference_dataset(ingame_unlabeled_reduced_dataset, model, args)


    ds = dataset_getter.get_outgame_dataset_cats("valid")
    ds2 = dataset_getter.get_outgame_dataset_cats(
        "valid", apply_augmentation=True
    )
    ds = tud.ConcatDataset([ds, ds2])
    (
        _,
        _,
        ys_subclass_if_outgame_valid,
        _,
        _,
        ps_subclass_if_outgame_valid,
        fts_outgame_valid,
    ) = inference_dataset(ds, model, args)
    ds = dataset_getter.get_outgame_dataset_cats("test")
    (
        _,
        _,
        ys_subclass_if_outgame_test,
        _,
        _,
        ps_subclass_if_outgame_test,
        fts_outgame_test,
    ) = inference_dataset(ds, model, args)


    ingame_labeled_dataset = tud.ConcatDataset(
        [
            # dataset_getter.get_labeled_ingame_dataset("train"),
            dataset_getter.get_labeled_ingame_dataset("valid"),
        ]
    )

    (
        ys_ingame,
        ys_whichgame,
        _,
        ps_ingame,
        ps_whichgame,
        _,
        fts,
    ) = inference_dataset(ingame_labeled_dataset, model, args)

    test_dataset = dataset_getter.get_labeled_ingame_dataset("test")
    (
        xs,
        ys_ingame_t,
        ys_whichgame_t,
        ys_subclass_if_outgame_t,
        ps_ingame_t,
        ps_whichgame_t,
        ps_subclass_if_outgame_t,
        fts_t,
    ) = inference_dataset(test_dataset, model, args, ret_just_one_x=True)


    ########################
    #    CALIBRATORS
    ########################
    
    calibrator_outgame_subclasses = train_calibrator(
        model.fc_outgame_subclasses, 
        fts_outgame_valid,
        ys_subclass_if_outgame_valid,
        fts_outgame_test,
        ys_subclass_if_outgame_test,
        args.num_outgame_classes,
        args,
        lr=0.01,
        weight_decay=1e-3, 
        type='outgame_subclasses')
    
    calibrator_ingame = train_calibrator(
        model.fc_ingame, 
        fts,
        ys_ingame,
        fts_t,
        ys_ingame_t,
        2,
        args,
        lr=0.01,
        weight_decay=1e-3, 
        type='ingame')

    calibrator_whichgame = train_calibrator(
        model.fc_game, 
        torch.cat([fts]),
        torch.cat([ys_whichgame]),
        fts_t,
        ys_whichgame_t,
        args.num_ingame_classes,
        args,
        lr=0.01,
        weight_decay=0.0, 
        type='whichgame')

    # Update the original model weighs with the calibrators weights 
    model.fc_outgame_subclasses.bias.data *= 0
    model.fc_outgame_subclasses.bias.data += calibrator_outgame_subclasses.fc.bias.data.to(
        model.fc_outgame_subclasses.bias.device
    )
    model.fc_outgame_subclasses.weight.data *= 0
    model.fc_outgame_subclasses.weight.data += calibrator_outgame_subclasses.fc.weight.data.to(
        model.fc_outgame_subclasses.weight.device
    )

    model.fc_game.bias.data *= 0
    model.fc_game.bias.data += calibrator_whichgame.fc.bias.data.to(
        model.fc_game.bias.device
    )
    model.fc_game.weight.data *= 0
    model.fc_game.weight.data += calibrator_whichgame.fc.weight.data.to(
        model.fc_game.weight.device
    )

    model.fc_ingame.bias.data *= 0
    model.fc_ingame.bias.data += calibrator_ingame.fc.bias.data.to(
        model.fc_ingame.bias.device
    )
    model.fc_ingame.weight.data *= 0
    model.fc_ingame.weight.data += calibrator_ingame.fc.weight.data.to(
        model.fc_ingame.weight.device
    )

    model.eval()
    def get_trace_path(info_str, run_name=args.experiment_name, commithash=get_commit_hash()):
        return f"{run_name}_{commithash}{info_str}.pth"

    # Save the classnames
    os.makedirs("calibrate_outs", exist_ok=True)
    os.makedirs("calibrate_outs/"+str(args.calibrator_name), exist_ok=True)
    with open(
        os.path.join("calibrate_outs", str(args.calibrator_name), get_trace_path("")[:-4])
        + "_classes.json",
        "w",
    ) as f:
        json.dump([i for i in args.ingame_classes], f)

    torch.save(
        model.cpu().eval().state_dict(),
        os.path.join("calibrate_outs", str(args.calibrator_name), get_trace_path("_calibrated")),
    )

    # Save the model trace (imagenet normalized version)
    trace = torch.jit.trace(WrapIt(model.cpu()), torch.randn(xs[:1].shape))
    trace_path = get_trace_path("")
    torch.jit.save(trace, os.path.join("calibrate_outs", str(args.calibrator_name), trace_path))

    # Save the model trace (RGB255 input version)
    trace = torch.jit.trace(WrapIt255(model.cpu()), torch.randn(xs[:1].shape))
    trace_path = get_trace_path("-rgb255in")
    torch.jit.save(trace, os.path.join("calibrate_outs", str(args.calibrator_name), trace_path))

    coreml_path = os.path.join(
        "calibrate_outs", str(args.calibrator_name),
        trace_path.replace("mdl", "coreml").replace(".pth", ".mlmodel"),
    )

    # Now ONNX export for coreml
    onnx_path = coreml_path[:-7] + "onnx"
    model.eval()
    m = WrapIt255(model.cpu())
    input_names = ["image"]
    output_names = ["out_game_in_game_proba", "game_proba", "nude_safe_proba"]
    with torch.no_grad():
        torch_out = torch.onnx.export(
            m,
            xs[:1],
            onnx_path,
            export_params=True,
            opset_version=7,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
        )

    with open("conv_dict.json", "r") as f:
        conv_dict = json.load(f)
    coreml_model = convert(
        model=onnx_path,
        image_input_names=input_names,
        minimum_ios_deployment_target="12",
    )
    coreml_model.author = "Liam Schoneveld"
    coreml_model.user_defined_metadata[
        "whichgame-ai commit hash"
    ] = get_commit_hash()
    coreml_model.user_defined_metadata["game_codes"] = ",".join(
        [conv_dict[k] for k in ingame_classes]
    )
    coreml_model.user_defined_metadata["outgame_classes"] = ",".join(
        outgame_classes
    )
    coreml_model.short_description = "TESTING version of whichgame-porn model, if in-game, detects between 79 games, if out-game, detects between porn/safe"
    coreml_model.save(coreml_path,)
    print("Saved coreml model to", coreml_path)

if __name__ == "__main__":
    args = whichgame_args().parse_args()
    train(args)
    
    

