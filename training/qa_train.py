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
from models.ITA_single_layer_upsample_shuffle.QAT.model import ITALSTMNetVIT_QAT
from third_party.vitfly_FPGA.training.train import TRAINER

class QATTrainer(TRAINER):
    def __init__(self, args):
        # 1. --- CORRECT INITIALIZATION ---
        # Call the parent TRAINER's __init__ method FIRST.
        # This is critical as it sets up the dataset, workspace, logging,
        # and all other attributes from 'args'. It also creates a float model
        # and an optimizer, which we will override next.
        super().__init__(args)

        # --- Now, override the model and optimizer for QAT ---
        self.mylogger("[QAT] Overriding float model with ITALSTMNetVIT_QAT...")
        self.model = ITALSTMNetVIT_QAT().float() # The QAT model starts as float

        # 2. --- LOAD PRE-TRAINED WEIGHTS (IF APPLICABLE) ---
        # This logic allows for fine-tuning a pre-trained float model.
        # If args.load_checkpoint_qat is False, this block is skipped,
        # enabling training a QAT model from scratch.
        if args.load_checkpoint_qat:
            self.mylogger(f"[QAT] Loading pre-trained float weights from: {args.checkpoint_path}")
            if not os.path.exists(args.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint_path}")
            ckpt = torch.load(args.checkpoint_path, map_location='cpu') # Load to CPU first
            self.model.load_state_dict(ckpt, strict=False)
            self.mylogger("[QAT] Float weights loaded successfully.")
        
        self.model.to(self.device) # Move the model to the target device

        # 3. --- CONFIGURE AND PREPARE MODEL FOR QAT ---
        self.mylogger("[QAT] Configuring model for Quantization-Aware Training...")
        
        # Set the backend engine. 'qnnpack' is for ARM, 'fbgemm' for x86.
        # It's good practice to set this based on the target deployment platform.
        torch.backends.quantized.engine = 'qnnpack'

        # Define a symmetric quantization configuration for activations and weights.
        # This is often required for hardware accelerators.
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
        
        # Apply the qconfig to the specific modules you intend to quantize.
        self.model.attention_blocks.qconfig = ita_symmetric_qconfig
        self.model.ffn_blocks.qconfig = ita_symmetric_qconfig
        
        self.mylogger("[QAT] Preparing model with fake quantization modules (observers)...")
        # prepare_qat inserts fake_quant modules into the model graph.
        # This must be done BEFORE creating the optimizer for the QAT run.
        torch.ao.quantization.prepare_qat(self.model, inplace=True)

        # 4. --- RE-INITIALIZE OPTIMIZER ---
        # The optimizer must be recreated *after* prepare_qat has modified the model,
        # so it can manage the parameters of the newly inserted observer modules.
        self.mylogger("[QAT] Re-initializing optimizer for the QAT-prepared model.")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def finalize(self):
        """Converts the trained QAT model to a fully quantized integer model and saves it."""
        self.mylogger("[QAT] Finalizing training and converting to a fully quantized integer model...")
        
        # The model must be on the CPU and in eval mode for conversion.
        self.model.to('cpu')
        self.model.eval()
        
        # The convert step fuses modules (e.g., Conv-BN-ReLU) and replaces float
        # operations with their integer counterparts.
        torch.ao.quantization.convert(self.model, inplace=True)
        
        out_path = os.path.join(self.workspace, "model_quantized_final.pth")
        torch.save(self.model.state_dict(), out_path)
        self.mylogger(f"[QAT] Fully quantized model saved to {out_path}")


def argparsing():
    """Parses command-line and config file arguments."""
    parser = configargparse.ArgumentParser()
    
    # Use uname to construct default paths dynamically
    uname = os.getlogin()

    # general params
    parser.add_argument('--config', is_config_file=True, help='config file relative path')
    parser.add_argument('--basedir', type=str, default=f'/home/{uname}/agile_ws/src/agile_flight', help='path to repo')
    parser.add_argument('--logdir', type=str, default='learner/logs', help='path to relative logging directory')
    parser.add_argument('--datadir', type=str, default=f'/home/{uname}/agile_ws/src/agile_flight', help='path to relative dataset directory')
    
    # experiment-level and learner params
    parser.add_argument('--ws_suffix', type=str, default='', help='suffix if any to workspace name')
    parser.add_argument('--model_type', type=str, default='ITALSTMNetVIT', help='This will be IGNORED by QATTrainer but is needed by the base TRAINER init.')
    parser.add_argument('--dataset', type=str, default='5-2', help='name of dataset')
    parser.add_argument('--short', type=int, default=0, help='if nonzero, how many trajectory folders to load')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of dataset to use for validation')
    parser.add_argument('--seed', type=int, default=None, help='random seed for python, numpy, and torch')
    parser.add_argument('--device', type=str, default='cuda', help='device to use (e.g., cuda, cpu)')
    
    # Checkpoint handling flags
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='DO NOT USE for QAT. Base trainer flag.')
    parser.add_argument('--load_checkpoint_qat', action='store_true', default=False, help='Load a float checkpoint to START QAT fine-tuning.')
    parser.add_argument('--checkpoint_path', type=str, default='', help='absolute path to model checkpoint')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--N_eps', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='number of epochs to warmup learning rate for')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='whether to use learning rate decay')
    parser.add_argument('--save_model_freq', type=int, default=25, help='frequency with which to save model checkpoints')
    parser.add_argument('--val_freq', type=int, default=10, help='frequency with which to evaluate on validation set')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Number of validation checks to wait for improvement before stopping. Set to 0 to disable.')

    args = parser.parse_args()
    if args.config:
        print(f'[CONFIGARGPARSE] Parsing args from config file {args.config}')

    return args

if __name__ == "__main__":
    # Parse arguments
    args = argparsing()
    print(args)

    # Instantiate the QAT-specific trainer
    trainer = QATTrainer(args)
    
    # Run the standard training loop from the parent class
    # This loop is now operating on the QAT-prepared model
    trainer.train()
    
    # After training, convert the model to its final integer form
    trainer.finalize()