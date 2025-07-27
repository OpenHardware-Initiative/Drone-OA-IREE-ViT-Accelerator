import os
import sys
import torch
import torch.quantization
import configargparse

# ensure project root is on Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from third_party.vitfly.training.train import TRAINER
from models.quantized.quant_ready_ITAConformerLSTM import QuantReadyITALSTM

class QATTrainer(TRAINER):
    def __init__(self, args):
        # Initialize dataloaders, workspace, etc.
        super().__init__(args)

        # Override model with quant-ready version
        self.model = QuantReadyITALSTM(itaparameters=None, efficient_attn=True) \
                         .to(self.device).float()

        # (Optional) load a float pretrained checkpoint
        if args.load_checkpoint_qat:
            ckpt = torch.load(args.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt, strict=False)

        # QAT configuration
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        self.model.fuse_model()  
        torch.quantization.prepare_qat(self.model, inplace=True)

        # Recreate optimizer for QAT
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # debug and see if the model is correctly initialized
        
        self.mylogger(f"[QAT] Model is QuantReadyITALSTM: {isinstance(self.model, QuantReadyITALSTM)}")

    def finalize(self):
        # Convert to a fully quantized model and save
        self.model.eval()
        torch.quantization.convert(self.model, inplace=True)
        out_path = os.path.join(self.workspace, "model_qat.pth")
        torch.save(self.model.state_dict(), out_path)
        self.mylogger(f"[QAT] Quantized model saved to {out_path}")

def argparsing():

    import configargparse
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
    parser.add_argument('--seed', type=int, default=None, help='random seed to use for python random, numpy, and torch -- WARNING, probably not fully implemented')
    parser.add_argument('--device', type=str, default='cuda', help='generic cuda device; specific GPU should be specified in CUDA_VISIBLE_DEVICES')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='whether to load from a model checkpoint')
    parser.add_argument('--load_checkpoint_qat', action='store_true', default=False, help='whether to load from a quant-ready model checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=f'/Users/denizonat/REPOS/neuroTUM/Drone-ViT-HW-Accelerator/models/pretrained_models/checkpoints_for_qat/ITALSTM.pth', help='absolute path to model checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--N_eps', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='number of epochs to warmup learning rate for')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='whether to use lr_decay, hardcoded to exponentially decay to 0.01 * lr by end of training')
    parser.add_argument('--save_model_freq', type=int, default=25, help='frequency with which to save model checkpoints')
    parser.add_argument('--val_freq', type=int, default=10, help='frequency with which to evaluate on validation set')


    args = parser.parse_args()
    print(f'[CONFIGARGPARSE] Parsing args from config file {args.config}')

    return args

if __name__ == "__main__":
    # force CUDA-tensors by default, if available
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # parse arguments (same as train.py)
    args = argparsing()

    # run QAT
    trainer = QATTrainer(args)
    trainer.train()      # runs standard train loop, but under QAT
    trainer.finalize()   # converts & saves the quantized model