import os
import sys
import torch
import torch.quantization
import configargparse

# ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from third_party.vitfly.training.train import TRAINER, argparsing
from models.quantized.quant_ready_ITAConformerLSTM import QuantReadyITALSTM

class QATTrainer(TRAINER):
    def __init__(self, args):
        # Initialize dataloaders, workspace, etc.
        super().__init__(args)

        # Override model with quant-ready version
        self.model = QuantReadyITALSTM(itaparameters=None, efficient_attn=True) \
                         .to(self.device).float()

        # (Optional) load a float pretrained checkpoint
        if args.load_checkpoint:
            ckpt = torch.load(args.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt, strict=False)

        # QAT configuration
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        self.model.fuse_model()  # no-op stub at top level / propagates to children
        torch.quantization.prepare_qat(self.model, inplace=True)

        # Recreate optimizer for QAT
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def finalize(self):
        # Convert to a fully quantized model and save
        self.model.eval()
        torch.quantization.convert(self.model, inplace=True)
        out_path = os.path.join(self.workspace, "model_qat.pth")
        torch.save(self.model.state_dict(), out_path)
        self.mylogger(f"[QAT] Quantized model saved to {out_path}")

if __name__ == "__main__":
    # force CUDA-tensors by default, if available
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # parse arguments (same as train.py)
    args = argparsing()

    # run QAT
    trainer = QATTrainer(args)
    trainer.train()      # runs standard train loop, but under QAT
    trainer.finalize()   # converts & saves the quantized model