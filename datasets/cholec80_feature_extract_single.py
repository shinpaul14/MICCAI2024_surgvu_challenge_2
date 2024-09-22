import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Resize, Normalize, ShiftScaleRotate
from albumentations.pytorch.transforms import ToTensorV2



class Cholec80FeatureExtract:
    def __init__(self, hparams):
        self.hparams = hparams
        self.dataset_mode = hparams.dataset_mode
        self.input_height = hparams.input_height
        self.input_width = hparams.input_width
        self.fps_sampling = hparams.fps_sampling
        self.fps_sampling_test = hparams.fps_sampling_test
        self.cholec_root_dir = Path(self.hparams.data_root)
        self.transformations = self.__get_transformations()

        self.df = {}
        self.df["all"] = pd.read_csv(self.cholec_root_dir / "surgvu_challnege.txt")

        self.current_fold = hparams.fold
        fold_map = {
            'train': ['case_145', 'case_072', 'case_151', 'case_048', 'case_139', 'case_067', 'case_121', 'case_154', 'case_096', 'case_065', 'case_051', 'case_138', 'case_039', 'case_014', 'case_087', 'case_077', 'case_075', 'case_112', 'case_128', 'case_046', 'case_041', 'case_068', 'case_010', 'case_143', 'case_012', 'case_131', 'case_038', 'case_045', 'case_058', 'case_064', 'case_025', 'case_137', 'case_115', 'case_082', 'case_098', 'case_027', 'case_100', 'case_001', 'case_083', 'case_089', 'case_086', 'case_003', 'case_002', 'case_023', 'case_122', 'case_117', 'case_004', 'case_026', 'case_111', 'case_053', 'case_015', 'case_084', 'case_130', 'case_009', 'case_006', 'case_126', 'case_109', 'case_103', 'case_102', 'case_142', 'case_050', 'case_133', 'case_018', 'case_016', 'case_059', 'case_043', 'case_030', 'case_124', 'case_114', 'case_123', 'case_092', 'case_028', 'case_036', 'case_090', 'case_113', 'case_125', 'case_148', 'case_062', 'case_129', 'case_152', 'case_037', 'case_136', 'case_057', 'case_091', 'case_108', 'case_040', 'case_097', 'case_032', 'case_022', 'case_127', 'case_088', 'case_008', 'case_031', 'case_005', 'case_104', 'case_060', 'case_056', 'case_061', 'case_110', 'case_044', 'case_116', 'case_150', 'case_013'],
            'val': ['case_052', 'case_079', 'case_049', 'case_066', 'case_149', 'case_076', 'case_106', 'case_011', 'case_063', 'case_029', 'case_070', 'case_073', 'case_118', 'case_055', 'case_095', 'case_078', 'case_146', 'case_035', 'case_069', 'case_007', 'case_019', 'case_042', 'case_119', 'case_101', 'case_107', 'case_000', 'case_034', 'case_099', 'case_081', 'case_094', 'case_135', 'case_017', 'case_080', 'case_141', 'case_020', 'case_054', 'case_134', 'case_144', 'case_132', 'case_105', 'case_085', 'case_120', 'case_047', 'case_147']
        }

        self.df["all"]["fold"] = -1
        video_idx = [int(x[5:]) for x in self.df["all"]['case']]
        self.df["all"]['video_idx'] = video_idx
        self.df["all"]['task'] = self.df["all"]['task'].astype(int)

        all_list = []
        for fold, video_list in fold_map.items():
            all_list.extend(video_list)
            self.df["all"].loc[self.df["all"]["case"].isin(video_list), "fold"] = fold
        self.df["all"]["fold"] = self.df["all"]["fold"].astype(str)

        trn_idx = self.df["all"][self.df["all"]["fold"] == 'train'].index
        val_idx = self.df["all"][self.df["all"]["fold"] == 'val'].index

        self.df["train"] = self.df["all"].loc[trn_idx].reset_index(drop=True)
        self.df["val"] = self.df["all"].loc[val_idx].reset_index(drop=True)
        self.df["test"] = self.df["all"].loc[self.df["all"]["case"].isin(all_list)].reset_index()
        self.df["test"]['fold'] = 'test'

        self.vids_for_training = self.df["train"]["video_idx"].unique()
        self.vids_for_val = self.df["val"]["video_idx"].unique()
        self.vids_for_test = self.df["test"]["video_idx"].unique()

        task_list = list(self.df["train"]['task'])
        task_weight = self.median_frequency_weights(task_list, num_classes=8)
        self.class_weights_task = np.asarray(task_weight)

        self.data = {}
        self.label_col = self.hparams.task
        for split in ["train", "val"]:
            self.data[split] = Dataset_from_Dataframe(
                self.df[split],
                self.transformations[split],
                self.label_col,
                img_root=self.cholec_root_dir / "data",
                image_path_col="frame",
                add_label_cols=["task"]
            )

        self.data["test"] = Dataset_from_Dataframe(
            self.df["test"],
            self.transformations["test"],
            self.label_col,
            img_root=self.cholec_root_dir / "data",
            image_path_col="frame",
            add_label_cols=["video_idx", "frame", "index", "task"]
        )

    def median_frequency_weights(self, file_list, num_classes):
        frequency = [0] * num_classes
        for label in file_list:
            frequency[label] += 1
        median = np.median([freq for freq in frequency if freq > 0])
        weights = [0 if freq == 0 else median / freq for freq in frequency]
        return weights

    def __get_transformations(self):
        norm_mean = [0.3456, 0.2281, 0.2233]
        norm_std = [0.2528, 0.2135, 0.2104]
        normalize = Normalize(mean=norm_mean, std=norm_std)
        training_augmentation = Compose([
            ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.2, 0.5), rotate_limit=15, border_mode=0, value=0, p=0.7)
        ])

        data_transformations = {}
        data_transformations["train"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            training_augmentation,
            normalize,
            ToTensorV2(),
        ])
        data_transformations["val"] = Compose([
            Resize(height=self.input_height, width=self.input_width),
            normalize,
            ToTensorV2(),
        ])
        data_transformations["test"] = data_transformations["val"]
        return data_transformations

    @staticmethod
    def add_dataset_specific_args(parser):
        cholec80_specific_args = parser.add_argument_group(
            title='cholec80 specific args options')
        cholec80_specific_args.add_argument("--fps_sampling", type=float, default=25)
        cholec80_specific_args.add_argument("--fps_sampling_test", type=float, default=25)
        cholec80_specific_args.add_argument("--dataset_mode", default='video', choices=['vid_multilabel', 'img', 'img_multilabel', 'img_multilabel_feature_extract'])
        cholec80_specific_args.add_argument("--test_extract", action="store_true")
        return parser


class Dataset_from_Dataframe(Dataset):
    def __init__(self, df, transform, label_col, img_root="", image_path_col="path", add_label_cols=[]):
        self.df = df
        self.transform = transform
        self.label_col = label_col
        self.image_path_col = image_path_col
        self.img_root = img_root
        self.add_label_cols = add_label_cols

    def __len__(self):
        return len(self.df)

    def load_from_path(self, index):
        img_path_df = self.df.loc[index, self.image_path_col]
        p = self.img_root / img_path_df
        X = Image.open(p)
        X_array = np.array(X)
        return X_array, p

    def __getitem__(self, index):
        X_array, p = self.load_from_path(index)
        if self.transform:
            X = self.transform(image=X_array)["image"]
        label = torch.tensor(int(self.df[self.label_col][index]))

        if len(self.add_label_cols) != 1:
            add_label = []
            for add_l in self.add_label_cols:
                add_label.append(self.df[add_l][index])
        else:
            add_label_cols = self.add_label_cols[0]
            add_label = torch.tensor(int(self.df[add_label_cols][index]))

        X = X.type(torch.FloatTensor)
        return X, label, add_label
