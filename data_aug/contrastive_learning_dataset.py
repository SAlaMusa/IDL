from torchvision import transforms, datasets
from data_aug.gaussian_blur import GaussianBlur
from data_aug.gaussian_noise import GaussianNoise
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, aug_cfg=None):
        """Build the SimCLR augmentation pipeline from an aug_cfg dict.

        If aug_cfg is None or empty, the full SimCLR baseline recipe is used:
            RandomResizedCrop → RandomHorizontalFlip → ColorJitter (p=0.8) →
            RandomGrayscale (p=0.2) → GaussianBlur → ToTensor

        aug_cfg keys (all default to the baseline value):
            crop        (bool, default True)  – RandomResizedCrop; False → Resize
            flip        (bool, default True)  – RandomHorizontalFlip
            jitter      (bool, default True)  – ColorJitter (p=0.8)
            jitter_strength (float, default 1.0) – s parameter for ColorJitter
            grayscale   (bool, default True)  – RandomGrayscale (p=0.2)
            blur        (bool, default True)  – GaussianBlur

            # Harmful augmentations (default False — not in baseline)
            rotation    (bool, default False) – RandomRotation
            rotation_degrees (int, default 180)
            solarize    (bool, default False) – RandomSolarize
            solarize_threshold (int, default 128)
            solarize_p  (float, default 0.5)
            noise       (bool, default False) – GaussianNoise (tensor, after ToTensor)
            noise_std   (float, default 0.3)
        """
        if aug_cfg is None:
            aug_cfg = {}

        crop      = aug_cfg.get('crop',      True)
        flip      = aug_cfg.get('flip',      True)
        jitter    = aug_cfg.get('jitter',    True)
        s         = aug_cfg.get('jitter_strength', 1.0)
        grayscale = aug_cfg.get('grayscale', True)
        blur      = aug_cfg.get('blur',      True)
        rotation  = aug_cfg.get('rotation',  False)
        solarize  = aug_cfg.get('solarize',  False)
        noise     = aug_cfg.get('noise',     False)

        pipeline = []

        # --- PIL-space transforms ---
        if crop:
            pipeline.append(transforms.RandomResizedCrop(size=size))
        else:
            pipeline.append(transforms.Resize(size))

        if flip:
            pipeline.append(transforms.RandomHorizontalFlip())

        if rotation:
            degrees = aug_cfg.get('rotation_degrees', 180)
            pipeline.append(transforms.RandomRotation(degrees=degrees))

        if jitter:
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            pipeline.append(transforms.RandomApply([color_jitter], p=0.8))

        if grayscale:
            pipeline.append(transforms.RandomGrayscale(p=0.2))

        if solarize:
            threshold = aug_cfg.get('solarize_threshold', 128)
            p         = aug_cfg.get('solarize_p', 0.5)
            pipeline.append(transforms.RandomSolarize(threshold=threshold, p=p))

        if blur:
            pipeline.append(GaussianBlur(kernel_size=int(0.1 * size)))

        # --- Convert to tensor ---
        pipeline.append(transforms.ToTensor())

        # --- Tensor-space transforms ---
        if noise:
            noise_std = aug_cfg.get('noise_std', 0.3)
            pipeline.append(GaussianNoise(std=noise_std))

        return transforms.Compose(pipeline)

    def get_dataset(self, name, n_views, aug_cfg=None):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(
                self.root_folder, train=True,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(32, aug_cfg), n_views),
                download=True),

            'stl10': lambda: datasets.STL10(
                self.root_folder, split='unlabeled',
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(96, aug_cfg), n_views),
                download=True),
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
