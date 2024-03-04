# By Maciej Wiatrak
# The number of workers was 8
# the GPUs were NVIDIA A100-SXM-80GB
# lamindb 0.67.3

import os

import torch
from lamindb.dev import MappedCollection
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from tap import Tap
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader


class MLPModel(LightningModule):
    """Simple MLP classification model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_hidden_layers: int,
        dropout: float = 0.2,
        lr: float = 0.001,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(dropout)
        )

        tower_layers = []
        for _ in range(n_hidden_layers):
            tower_layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.mlp_tower = nn.Sequential(*tower_layers)

        self.output_layer = nn.Linear(hidden_size, output_size)
        self.n_classes = output_size
        self.lr = lr

        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        """Forward pass."""
        # pass through layers
        x = self.input_layer(x)
        x = self.mlp_tower(x)
        x = self.output_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        """Training step."""
        # get input to the model
        x = batch[0]
        logits = self(x)

        # create dummy labels
        labels = torch.randint(low=0, high=self.n_classes, size=(x.shape[0],))
        loss = self.loss_fn(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # get input to the model
        x = batch[0]
        logits = self(x)

        # create dummy labels
        labels = torch.randint(low=0, high=self.n_classes, size=(x.shape[0],))
        _ = self.loss_fn(logits, labels)

    def configure_optimizers(self):
        """Configure optimizers."""
        return AdamW(
            [param for param in self.parameters() if param.requires_grad],
            lr=self.lr,
        )


def run(
    input_dir: str,
    n_layers: int,
    input_size: int,
    hidden_size: int,
    output_size: int,
    dropout: float,
    batch_size: int,
    num_workers: int,
    num_epochs: int,
):
    """Run the training script."""
    dataset = MappedCollection(
        path_list=[os.path.join(input_dir, item) for item in os.listdir(input_dir) if item.endswith(".h5ad")],
        join=None,
        parallel=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    model = MLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_hidden_layers=n_layers,
        dropout=dropout,
    )

    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    trainer = Trainer(
        accelerator="cpu" if not torch.cuda.is_available() else "auto",
        strategy=strategy,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        max_epochs=num_epochs,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(model, dataloader)


class ArgParser(Tap):
    """Arguments training the model."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    input_dir: str = "/tmp/multi-gpu-test"
    n_layers: int = 5
    input_size: int = 20000
    hidden_size: int = 8192
    output_size: int = 10
    dropout: float = 0.2
    batch_size: int = 256
    num_workers: int = 8
    num_epochs: int = 100


if __name__ == "__main__":
    args = ArgParser().parse_args()
    run(
        input_dir=args.input_dir,
        n_layers=args.n_layers,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
    )
