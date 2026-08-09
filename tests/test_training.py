import torch
from torch.utils.data import TensorDataset, DataLoader
import pytest

from duobit.config import DuobitConfig
from duobit.model.transformer import DuobitTransformer
from duobit.training.trainer import Trainer

def test_full_training_pipeline_integration():
    torch.manual_seed(42)
    
    config = DuobitConfig(
        vocab_size=100,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        max_seq_len=32,
        group_size=64,
        lr=1e-3,
    )

    model = DuobitTransformer(config, use_duobit=True)
    
    # Generate synthetic text data
    input_ids = torch.randint(0, config.vocab_size, (64, config.max_seq_len))
    dataset = TensorDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=dataloader,
        device="cpu",
    )
    
    # Train for 20 steps
    results = trainer.train(max_steps=20, eval_freq=0)
    
    assert len(results["train_logs"]) > 0
    final_loss = results["train_logs"][-1]["loss"]
    assert final_loss > 0.0 and not torch.isnan(torch.tensor(final_loss))
    
    # Verify transition rate was measured
    assert "transition_rate" in results["train_logs"][-1]
