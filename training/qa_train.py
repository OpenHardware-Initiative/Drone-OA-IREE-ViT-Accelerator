import os
import sys
import torch
import torch.quantization
import configargparse

# Ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import our final model and the base trainer
from models.testing.ITA_model import ITALSTMNetVIT
from third_party.vitfly.training.train import TRAINER

class QATTrainer(TRAINER):
    def __init__(self, args):
        # Initialize dataloaders, workspace, etc. from the base TRAINER
        super().__init__(args)

        # --- IMPORTANT: Define hardware requantization parameters ---
        # These are placeholders. For a true hardware match, these would be
        # determined from post-training calibration or a more advanced QAT process.
        itaparameters = {
            "mq": 1.0, "sq": 0, "mk": 1.0, "sk": 0, "mv": 1.0, "sv": 0,
            "ma": 1.0, "sa": 0, "mav": 1.0, "sav": 0, "mo": 1.0, "so": 0,
            "m_ff1": 1.0, "s_ff1": 0, "m_ff2": 1.0, "s_ff2": 0,
            # ITAGELU params (used in non-QAT mode)
            "q_1": -22, "q_b": -14, "q_c": 24, "gelu_rqs_mul": 119, 
            "gelu_rqs_shift": 20, "gelu_rqs_add": 0,
        }

        # --- Override model with our quant-ready version in QAT mode ---
        self.mylogger("[QAT] Initializing ITALSTMNetVIT for QAT...")
        self.model = ITALSTMNetVIT(params=itaparameters, qat_mode=True) \
                         .to(self.device).float()

        # (Optional) load a float pretrained checkpoint
        if args.load_checkpoint_qat:
            self.mylogger(f"[QAT] Loading float weights from: {args.checkpoint_path}")
            ckpt = torch.load(args.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt, strict=False)

        # --- Standard QAT setup process ---
        self.mylogger("[QAT] Configuring model for Quantization-Aware Training...")
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') # 'fbgemm' for x86, 'qnnpack' for ARM
        
        self.mylogger("[QAT] Fusing model layers...")
        self.model.fuse_model()
        
        self.mylogger("[QAT] Preparing model with observers...")
        torch.quantization.prepare_qat(self.model, inplace=True)

        # Recreate optimizer after model modification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def finalize(self):
        """Converts the trained QAT model to a fully quantized integer model and saves it."""
        self.mylogger("[QAT] Finalizing training and converting to integer model...")
        self.model.eval()
        torch.quantization.convert(self.model, inplace=True)
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