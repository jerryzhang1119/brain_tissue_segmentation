from datasets.drive_dataset import DRIVEDataset, DRIVETestDataset, DRIVEMetricDataset
from datasets.chasedb1_dataset import ChaseDB1Dataset, ChaseDB1TestDataset, ChaseDB1MetricDataset

def get_dataset(dataset_name, part):
    dataset_dict = {
        'CHASEDB1': {
            'train': ChaseDB1Dataset(
                data_path='datasets/train/images',
                label_path='datasets/train/gt',
                edge_path='datasets/train/edges',
                need_enhance=False
            ),
            'val': ChaseDB1Dataset(
                data_path='datasets/val/images',
                label_path='datasets/val/gt',
                edge_path='datasets/val/edges',
                need_enhance=False
            ),
            'test': ChaseDB1TestDataset(
                data_path='datasets/val/images',
            ),
            'metric': ChaseDB1MetricDataset(
                data_path='datasets/val/images',
                label_path='datasets/val/gt',
                edge_path='datasets/val/edges',
            ),
        }
    }
    return dataset_dict[dataset_name][part]
