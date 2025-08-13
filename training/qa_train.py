import os
import sys
import torch
import torch.quantization
import configargparse
from torch.quantization import QConfig, FusedMovingAvgObsFakeQuantize


# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import our final model and the base trainer
from models.ITA.QAT.model import ITALSTMNetVIT_QAT
from third_party.vitfly_FPGA.training.train import TRAINER

class QATTrainer(TRAINER):
    def __init__(self, args):
        # Initialize dataloaders, workspace, etc. from the base TRAINER
        super().__init__(args)

        # --- Override model with our quant-ready version in QAT mode ---
        self.mylogger("[QAT] Initializing ITALSTMNetVIT for QAT...")
        self.model = ITALSTMNetVIT_QAT().to(self.device).float()

        # (Optional) load a float pretrained checkpoint
        if args.load_checkpoint_qat:
            self.mylogger(f"[QAT] Loading float weights from: {args.checkpoint_path}")
            ckpt = torch.load(args.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt, strict=False)

        # --- QAT setup with HARDWARE-COMPLIANT symmetric qconfig ---
        self.mylogger("[QAT] Configuring model for Quantization-Aware Training...")
        
        torch.backends.quantized.engine = 'qnnpack'

        ita_symmetric_qconfig = QConfig(
            activation=FusedMovingAvgObsFakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric, reduce_range=False
            ),
            weight=FusedMovingAvgObsFakeQuantize.with_args(
                observer=torch.quantization.MovingAverageMinMaxObserver,
                quant_min=-128, quant_max=127, dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric, reduce_range=False
            )
        )
        
        self.model.attention_blocks.qconfig = ita_symmetric_qconfig
        self.model.ffn_blocks.qconfig = ita_symmetric_qconfig
        
        self.mylogger("[QAT] Preparing model with observers...")
        torch.ao.quantization.prepare_qat(self.model, inplace=True)

        # Recreate optimizer after model modification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def finalize(self):
        """Converts the trained QAT model to a fully quantized integer model and saves it."""
        self.mylogger("[QAT] Finalizing training and converting to integer model...")
        self.model.cpu()
        self.model.eval()
        torch.ao.quantization.convert(self.model, inplace=True)
        out_path = os.path.join(self.workspace, "model_quantized_final.pth")
        torch.save(self.model.state_dict(), out_path)
        self.mylogger(f"[QAT] Fully quantized model saved to {out_path}")

def argparsing():
    """Parses command-line and config file arguments."""
    parser = configargparse.ArgumentParser()

    # general params
    parser.add_argument('--config', is_config_file=True, help='config file relative path')
    parser.add_argument('--basedir', type=str, default=f'', help='path to repo')
    parser.add_argument('--logdir', type=str, default='learner/logs', help='path to relative logging directory')
    parser.add_argument('--datadir', type=str, default=f'', help='path to relative dataset directory')
    
    # experiment-level and learner params
    parser.add_argument('--ws_suffix', type=str, default='', help='suffix if any to workspace name')
    parser.add_argument('--model_type', type=str, default='LSTMNet', help='string matching model name in lstmArch.py')
    parser.add_argument('--dataset', type=str, default='5-2', help='name of dataset')
    parser.add_argument('--short', type=int, default=0, help='if nonzero, how many trajectory folders to load')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of dataset to use for validation')
    parser.add_argument('--seed', type=int, default=None, help='random seed for python, numpy, and torch')
    parser.add_argument('--device', type=str, default='cuda', help='device to use (e.g., cuda, cpu)')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='whether to load from a model checkpoint')
    parser.add_argument('--load_checkpoint_qat', action='store_true', default=False, help='whether to load a float checkpoint to start QAT')
    parser.add_argument('--checkpoint_path', type=str, default='', help='absolute path to model checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--N_eps', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='number of epochs to warmup learning rate for')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='whether to use learning rate decay')
    parser.add_argument('--save_model_freq', type=int, default=25, help='frequency with which to save model checkpoints')
    parser.add_argument('--val_freq', type=int, default=10, help='frequency with which to evaluate on validation set')

    args = parser.parse_args()
    print(f'[CONFIGARGPARSE] Parsing args from config file {args.config}')

    return args

if __name__ == "__main__":
    # Parse arguments
    args = argparsing()

    # Instantiate the QAT-specific trainer
    trainer = QATTrainer(args)
    
    # Run the standard training loop (now with a QAT-enabled model)
    trainer.train()
    
    # Convert the model to a final integer version
    trainer.finalize()