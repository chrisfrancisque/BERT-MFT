from dataclasses import dataclass
import os

@dataclass
class GradientAnalysisConfig:
    """Configuration for this mask fine-tuning experiment"""

    # Model Settings
    model_name: str = 'bert-base-uncased'
    num_labels: int = 2

    # Dataset Settings
    dataset_name: str = 'sst2'
    max_seq_length: int = 128
    train_samples: int = 1000

    # Training Settings for gradient colleciton
    batch_size: int = 32
    learning_rate: float = 2e-5
    detrimental_threshold_percentile: int = 25

    # TPU settings
    tpu_num_cores: int = 8
    use_tpu: bool = False

    #Output settings
    output_dir: str = './results'
    save_gradients: bool = True

    # Environment
    device: str = 'cpu'

    def __post_init__(self):
        """Creates output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def total_batch_size(self):
        """Create total batch size based on TPU usage"""
        if self.use_tpu:
            return self.batch_size * self.tpu_num_cores
        return self.batch_size
    
    @property
    def steps_per_epoch(self):
        """Calculate number of steps in one epoch"""
        return self.train_samples // self.batch_size
    
config = GradientAnalysisConfig()
