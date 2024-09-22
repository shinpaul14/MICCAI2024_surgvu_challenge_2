from pathlib import Path
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd


class Cholec80Helper(Dataset):
    def __init__(self, hparams, data_p, dataset_split=None):
        assert dataset_split != None
        self.data_p = data_p
        assert hparams.data_root != ""
    
        name = hparams.teacher_exp_name
        self.data_root = hparams.data_root
        self.data_root = Path(hparams.data_root).absolute() / name / 'pickle_export'
        self.number_vids = len(self.data_p)
        self.dataset_split = dataset_split
        self.factor_sampling = hparams.factor_sampling
 

    def __len__(self):
        return self.number_vids

    def __getitem__(self, index):
        vid_id = index
        p = self.data_root / self.data_p[vid_id]
 
        unpickled_x = pd.read_pickle(p)
    
        stem = np.asarray(unpickled_x[0],
                          dtype=np.float32)[::self.factor_sampling]
        y_hat = np.asarray(unpickled_x[1],
                           dtype=np.float32)[::self.factor_sampling]
        y = np.asarray(unpickled_x[2])[::self.factor_sampling]

        return stem, y_hat, y


class Cholec80():
    def __init__(self, hparams):
        # self.name = "Cholec80Pickle"
        self.hparams = hparams
        self.out_features = self.hparams.out_features
        self.features_per_seconds = hparams.features_per_seconds
        hparams.factor_sampling = 1 #(int(25 / hparams.features_subsampling))

        self.current_fold = hparams.fold
#         fold_map = {
#             "train":  ['case_145', 'case_072', 'case_151', 'case_048', 'case_139', 'case_067', 'case_121', 'case_154', 'case_096', 'case_065', 'case_051', 'case_138', 'case_039', 'case_014', 'case_087', 'case_077', 'case_075', 'case_112', 'case_128', 'case_046', 'case_041', 'case_068', 'case_010', 'case_143', 'case_012', 'case_131', 'case_038', 'case_045', 'case_058', 'case_064', 'case_025', 'case_137', 'case_115', 'case_153', 'case_082', 'case_098', 'case_027', 'case_100', 'case_001', 'case_083', 'case_089', 'case_086', 'case_003', 'case_002', 'case_093', 'case_023', 'case_122', 'case_117', 'case_004', 'case_026', 'case_111', 'case_053', 'case_015', 'case_084', 'case_130', 'case_009', 'case_006', 'case_126', 'case_109', 'case_103', 'case_102', 'case_142', 'case_050', 'case_133', 'case_018', 'case_016', 'case_059', 'case_043', 'case_030', 'case_124', 'case_024', 'case_114', 'case_123', 'case_092', 'case_028', 'case_036', 'case_090', 'case_113', 'case_125', 'case_148', 'case_062', 'case_129', 'case_152', 'case_037', 'case_136', 'case_057', 'case_091', 'case_108', 'case_040', 'case_097', 'case_032', 'case_022', 'case_074', 'case_127', 'case_088', 'case_008', 'case_031', 'case_005', 'case_104', 'case_060', 'case_056', 'case_061', 'case_110', 'case_044', 'case_116', 'case_150', 'case_013'],
#             "val": ['case_052', 'case_079', 'case_049', 'case_066', 'case_149', 'case_076', 'case_106', 'case_011', 'case_063', 'case_029', 'case_070', 'case_073', 'case_118', 'case_055', 'case_095', 'case_078', 'case_146', 'case_035', 'case_069', 'case_007', 'case_019', 'case_042', 'case_119', 'case_101', 'case_107', 'case_000', 'case_034', 'case_099', 'case_081', 'case_094', 'case_135', 'case_017', 'case_021', 'case_080', 'case_141', 'case_033', 'case_020', 'case_054', 'case_134', 'case_144', 'case_132', 'case_105', 'case_085', 'case_120', 'case_140', 'case_047', 'case_147'],
        fold_map= {'train': ['case_145', 'case_072', 'case_151', 'case_048', 'case_139', 'case_067', 'case_121', 'case_154', 'case_096', 'case_065', 'case_051', 'case_138', 'case_039', 'case_014', 'case_087', 'case_077', 'case_075', 'case_112', 'case_128', 'case_046', 'case_041', 'case_068', 'case_010', 'case_143', 'case_012', 'case_131', 'case_038', 'case_045', 'case_058', 'case_064', 'case_025', 'case_137', 'case_115', 'case_082', 'case_098', 'case_027', 'case_100', 'case_001', 'case_083', 'case_089', 'case_086', 'case_003', 'case_002', 'case_023', 'case_122', 'case_117', 'case_004', 'case_026', 'case_111', 'case_053', 'case_015', 'case_084', 'case_130', 'case_009', 'case_006', 'case_126', 'case_109', 'case_103', 'case_102', 'case_142', 'case_050', 'case_133', 'case_018', 'case_016', 'case_059', 'case_043', 'case_030', 'case_124', 'case_114', 'case_123', 'case_092', 'case_028', 'case_036', 'case_090', 'case_113', 'case_125', 'case_148', 'case_062', 'case_129', 'case_152', 'case_037', 'case_136', 'case_057', 'case_091', 'case_108', 'case_040', 'case_097', 'case_032', 'case_022', 'case_127', 'case_088', 'case_008', 'case_031', 'case_005', 'case_104', 'case_060', 'case_056', 'case_061', 'case_110', 'case_044', 'case_116', 'case_150', 'case_013'],
                    'val': ['case_052', 'case_079', 'case_049', 'case_066', 'case_149', 'case_076', 'case_106', 'case_011', 'case_063', 'case_029', 'case_070', 'case_073', 'case_118', 'case_055', 'case_095', 'case_078', 'case_146', 'case_035', 'case_069', 'case_007', 'case_019', 'case_042', 'case_119', 'case_101', 'case_107', 'case_000', 'case_034', 'case_099', 'case_081', 'case_094', 'case_135', 'case_017', 'case_080', 'case_141', 'case_020', 'case_054', 'case_134', 'case_144', 'case_132', 'case_105', 'case_085', 'case_120', 'case_047', 'case_147']
        }
       
   
        for fold, video_list in fold_map.items():
         
            
            if fold=='val':
                val_idx = [int(x[5:]) for x in fold_map[fold]]
            else:
                train_idx= [int(x[5:]) for x in fold_map[fold]]
        test_idx = train_idx + val_idx
      
      



        self.data_p = {}
        self.data_p["train"] = [(
            f"case_{i:03d}_predictions.pkl"
        ) for i in train_idx]
        self.data_p["val"] = [(
            f"case_{i:03d}_predictions.pkl"
        ) for i in val_idx]
        self.data_p["test"] = [(
            f"case_{i:03d}_predictions.pkl"
        ) for i in val_idx]



        self.data = {}
        for split in ["train", "val", "test"]:
            self.data[split] = Cholec80Helper(hparams,
                                              self.data_p[split],
                                              dataset_split=split)

        print(
            f"train size: {len(self.data['train'])} - val size: {len(self.data['val'])} - test size:"
            f" {len(self.data['test'])}")
        

    def median_frequency_weights(
              self,  file_list, num_classes):  ## do only once and define weights in class
            # num_classes = len(set(file_list))
            frequency = [0] * num_classes
            # for i in file_list:
            #     frequency[int(i[1])] += 1
            for label in file_list:
                    frequency[label] += 1
            median = np.median(frequency)
            weights = [median / j for j in frequency]
            return weights

    @staticmethod
    def add_dataset_specific_args(parser):  # pragma: no cover
        cholec80_specific_args = parser.add_argument_group(
            title='cholec80 dataset specific args options')
        cholec80_specific_args.add_argument("--features_per_seconds",
                                                  default=1,
                                                  type=float)
        cholec80_specific_args.add_argument("--features_subsampling",
                                                  default=1,
                                                  type=float)

        return parser
