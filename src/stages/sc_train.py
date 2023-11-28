from pathlib import Path
import sys 
src_dir = Path.cwd()
sys.path.append(str(src_dir))
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from src.sc_classifier.config.core import config
from src.sc_classifier.trainer import Trainer


def sc_train():
    trainer = Trainer(
            loss_function=CrossEntropyLoss() , 
            optimizer=AdamW,
            load_data=True
    )
    trainer.train(dict(config.train_args))
    
if __name__ == "__main__":
    sc_train()