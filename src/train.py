import argparse
import time
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from argparse import Namespace
from pathlib import Path
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from architectures import ARCHS
from datamodule import AceriDataModule
from parser_sanitizers import str2bool, ckpt_parse
from lightningmodule import TrainingModule
import pickle


def main(args: Namespace) -> None:
    # fixe le random seed. Doit être fait de manière spécifique dans datamodule
    if args.seed is not None:
        pl.seed_everything(args.seed)

    # Assemblage du chekpoint name
    save_dir = Path.cwd() / ".." / "results" / args.project

    #formating to str output
    train_transforms = args.transforms["train"]
    if train_transforms is not None:
        train_transforms = tuple(train_transforms.keys())

    file_stem = f"{args.arch} y={args.target_codes} c={args.channels} t={train_transforms} " \
                f"{args.current_split}-{args.n_splits} s={args.seed} eps={args.smooth_eps:.1f}"
    chk_name = file_stem + "_{val_loss:.3E}_{val_f1:.4f}_{val_acc:.4f}"

    # Callbacks utilisés par le trainer
    early_stopper = EarlyStopping(monitor=args.target_metric,
                                  patience=args.early_stop_patience,
                                  verbose=True,
                                  mode="max", )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    my_logger = TensorBoardLogger(str(save_dir), name=args.project)
    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          filename=chk_name,
                                          verbose=True,
                                          save_top_k=1,
                                          monitor=args.target_metric,
                                          mode="max",
                                          period=1)

    print("Setting up data...")
    print(f"Current split is {args.current_split} out of {args.n_splits}")
    dm = AceriDataModule(**vars(args))
    dm.prepare_data()
    dm.setup("fit")  # doit être executé pour savoir le nombre de classe

    if args.load_this is not None:
        print(f"\nLoading {args.load_this}...")
        model = TrainingModule.load_from_checkpoint(args.load_this, **vars(args))
    else:
        print("\nInitiating model...")
        model = TrainingModule(labels=dm.labels,
                               tab_size=dm.tab_size,
                               num_classes=dm.num_classes,
                               **vars(args))

    model.hparams.update(dm.my_hparams)


    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=my_logger,
                                            callbacks=[
                                                early_stopper,
                                                lr_monitor,
                                                checkpoint_callback
                                            ],
                                            precision=16,
                                            gpus=1,
                                            benchmark=True,
                                            )
    if args.fit:
        if args.auto_lr_find:
            trainer.tune(model, datamodule=dm)
        trainer.fit(model, datamodule=dm)

    if args.evaluate:
        model.setup("test")
        trainer.test(model, datamodule=dm)

        if args.save_test_preds:
            """#put model in evaluation mope
            model.eval()
            #get all test data
            
            #if model is cuda, input need to be cuda
            x = [tensor.to(model.device) for tensor in x]"""
            #extract useful data
            _, y_dict = dm.test_ds[:]
            #inference time
            #y_hat = model(x)
            all_full_truth = dm.test_ds.dataset.full_truth
            all_ids = dm.test_ds.dataset.ids
            full_truth_index_mask = np.in1d(all_ids, y_dict["ids"], assume_unique=True)
            full_truth = all_full_truth[full_truth_index_mask]

            #save it
            pd.DataFrame(data={"full_truth": full_truth,
                               "truth": y_dict["targets"],
                               "The rest": model.test_preds[:, 0],
                               "OK": model.test_preds[:, 1]},
                         index=y_dict["ids"]).to_csv(str(save_dir / file_stem) + ".csv")



    if args.to_onnx:
        tracer = [torch.randn((1, model.num_channels, 446)),
                  torch.randn(1, model.tab_size)]
        with open(f"{save_dir / file_stem}.scaler", "wb") as handle: #todo make better scaler saving logic
            pickle.dump(dm.scalers, handle)
        model.to_onnx(f"{save_dir/file_stem}.onnx",
                      input_sample=tracer,
                      export_params=True,
                      input_names=["data_spectra", "data_tabular"],
                      output_names=["-".join(model.labels)])

    """    
    del dm
    del model.model
    del model
    del trainer
    torch.cuda.empty_cache()
    if torch.cuda.max_memory_allocated() != 0:
        warnings.warn("VRAM leak detected! A tensor somewhere was forgotten...")"""


def cli_main():
    # ------------
    # args
    # ------------
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=42,
                               metavar="[n]",
                               help="Random seed utilisé pour le split des données, l'initialisation des poids du "
                                    "modèle et les distributions utilisées lors du DataAugmentation. Défaut: 42")
    parent_parser.add_argument("--project",
                               type=str,
                               default="test",
                               metavar="[nom du projet]",
                               help="Nom du projet. Va générer un dossier à ce nom pour la sauvegarde des modèles et "
                                    "des résultats TensorBoard/feuille de résultats. Défaut: test")
    parent_parser.add_argument("--fit",
                               type=str2bool,
                               default=True,
                               metavar="{True, False}",
                               help="Booléen. Si vrai un entrainement va être fait du modèle, soit-il nouveau ou "
                                    "spécifié par --load_this. Défaut: True")
    parent_parser.add_argument("--evaluate",
                               type=str2bool,
                               default=True,
                               metavar="{True, False}",
                               help="Booléen. Si vrai va évaluer le modèle avec la division test spécifiée. "
                                    "Défaut: True")
    parent_parser.add_argument("--target_metric",
                               type=str,
                               default="val_f1",
                               metavar="[val_f1, val_acc, val_loss]",
                               help="Métrique cible utilisé pour la décision de descendre le taux d'apprentissage et "
                                    "pour l'enregistrement des meilleurs modèles. Défaut: val_f1")
    parent_parser.add_argument("--load_this",
                               type=ckpt_parse,
                               default=None,
                               metavar="[None, chemin de fichier]",
                               help="Si n'est pas None, charge le fichier de modèle .ckpt pour l'utiliser. "
                                    "Défaut: None")
    parent_parser.add_argument("--to_onnx",
                               type=str2bool,
                               default=False,
                               metavar="{True, False}",
                               help="Booléen. Si vrai va exporter le modèle sous format .onnx (très portable!)"
                                    "Défaut: False")
    parent_parser.add_argument("--save_test_preds", type=str2bool, default=False)  # todo

    og_parser = TrainingModule.add_model_specific_args(parent_parser)
    og_parser = AceriDataModule.add_datamodule_specific_args(og_parser)

    old_file = Path(__file__).parents[1] / 'datasets' / "30k_cleaned.pkl"
    chk = Path(r"D:\Prog\Projects\aCeriNN\project\results\big_test3") / "ResNetV2 y=['OVA', 'OK'] c=('277', '380', '425') t=('da',) 0-20 s=42 eps=0.0_val_loss=3.696E-01_val_f1=0.8394_val_acc=0.8394.ckpt"
    args = og_parser.parse_args(["--arch=ResNetV2"])  # I need to know which architecture first
    parser = ARCHS[args.arch].add_arch_specific_args(og_parser)  # gets architecture defaults params
    # override them here if wanted
    args = parser.parse_args([f"--train_source={str(old_file)}",
                              f"--project=big_test_resnet1013",
                              f"--arch=ResNetV2",
                              f"--n_splits=20",
                              f"--wf=1",
                              f"--block_units=3_4_23_3",
                              f"--bs=320",
                              f"--fit=True",
                              f"--to_onnx=False",
                              f"--num_workers=1",
                              f"--save_test_preds=True"
                              ])
    

    main(args)


if __name__ == '__main__':
    cli_main()
