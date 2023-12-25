from pathlib import Path
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .preprocessing_funcs import load_dataloaders
from .train_funcs import load_state, load_results, evaluate_, evaluate_results
from .infer import infer_from_trained
from .misc import load_pickle
from transformers import AutoConfig
from src.utils.loss import NCEandMAE
from .re_transfomers.re_trf import RE_Transformers
import time
import logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger("__file__")
inverse_dict = {"supplier": "customer", "customer": "supplier", "other": "other"}
inferer = infer_from_trained(detect_entities=False)

def train_and_fit(args):
    """Train and fit model
    this function designed to recieve many options to train pretrained model,

    """
    # identify the src_dir
    src_dir = Path(args.get('src_dir', './'))
    # define the initial shuffle state
    shuffle = False
    
    ##Using Float points 16
    if args["fp16"]:
        from apex import amp
    else:
        amp = None
    ## Detect CUDA
    cuda = torch.cuda.is_available()
    ## Get checkpoint and load in generator(dataloader)
    (
        train_loader,
        valid_loader,
        train_len,
        valid_len,
        rm,
        label_binarizer,
        weights
    ) = load_dataloaders(args, shuffle=shuffle)
    n_classes = len(rm.rel2idx.keys())

    logger.info("Loaded Relations Mapper with Keys {}.".format(rm.rel2idx.keys()))
    logger.info("Loaded %d Training samples." % train_len)
    logger.info("Loaded %d Validation samples." % valid_len)

    # Define model path and other required directories to save model artifacts
    model_path = src_dir / args["model_path"]
    # Create dir if not exist
    model_path.mkdir(parents=True, exist_ok=True)
    # Model path
    model_artifacts = model_path / "model.pth.tar"
    # Determine model config values
    model = args["model_size"]
    #Load BaseModel config
    config = AutoConfig.from_pretrained(model)
    # Set model configurations
    config.base_model = args["model_size"]
    config.num_labels = rm.n_classes
    config.label2id = rm.rel2idx
    config.id2label = rm.idx2rel
    # Initialize the model
    net  = RE_Transformers(config, load_base=True)
    # Load tokenizer from assets
    
    tokenizer_path = src_dir /  "artifacts/assets/{}_tokenizer.pkl".format(model)
    tokenizer = load_pickle(tokenizer_path)
    # Resize token embeddings
    net.model.resize_token_embeddings(len(tokenizer))
    e1_id = tokenizer.convert_tokens_to_ids("[E1]")
    e2_id = tokenizer.convert_tokens_to_ids("[E2]")
    assert e1_id != e2_id != 1
    inferer.tokenizer = tokenizer
    inferer.e1_id = e1_id
    inferer.e2_id = e2_id
    inferer.label2id = rm.rel2idx
    inferer.id2label = rm.idx2rel
    inferer.batch_size = args["batch_size"]
    # Load the latest checkpoint to continue training if required
    if cuda:
        net.cuda()
    start_epoch, best_pred, amp_checkpoint = load_state(
        net=net,
        model_path=model_path,
        optimizer=None,
        scheduler=None,
        load_best=args["update_ext"],
    )
    # 
    # Based on the certain parameters we pick the reuqired cost function
    # Available loss functions:
    # * Weighted Loss
    # * Noise Aware Loss
    #
    if args.get('weight_loss'):
        weights =  weights.cuda() if cuda else weights
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weights)
        logger.info("Using Weighted Loss")
    elif args.get("noise_aware_loss"):
        logger.info("Using NoiseAware Loss")
        criterion = NCEandMAE( alpha=1.0, beta=1.0, num_classes=n_classes)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        logger.info("Not Using Weighted Loss")        
    optimizer = optim.Adam(
        [{"params": net.parameters(), "lr": args["lr"]}],
        weight_decay=args["weight_decay"],
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30], gamma=0.8
    )
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=len(len))
    logger.info(f"start_epoch : {start_epoch}\tbest_pred : {best_pred}\t")
    if (args["fp16"]) and (amp is not None):
        logger.info("Using fp16...")
        net, optimizer = amp.initialize(net, optimizer, opt_level="O2")
        if amp_checkpoint is not None:
            amp.load_state_dict(amp_checkpoint)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[2, 4, 6, 8, 12, 15, 18, 20, 22, 24, 26, 30],
            gamma=0.8,
        )
        
    # Define the logs columns
    logs_columns = [
        "train_losses",
        "train_accuracy",
        "valid_accuracy",
        "valid_f1_macro",
        "valid_recall",
        "valid_precision",
    ] + [f"{k}_f1_score" for k in net.config.label2id.keys()]
    # Load previous logs if exists
    all_logs = (
        load_results(model_path=model_path, logs_columns=logs_columns)
        if args["update_ext"]
        else [])
    
    logger.info("Starting training process...")
    pad_id = tokenizer.pad_token_id
    # define number of steps to log training progress
    update_size = len(train_loader) // 10
    for epoch in range(start_epoch, args["num_epochs"]):
        start_time = time.time()
        ## Set the network to train mode and define the metrics that need to be monitored.
        net.train()
        total_loss = 0.0
        losses_per_batch = []
        total_acc = 0.0
        accuracy_per_batch = []
        for i, data in enumerate(train_loader, 0):
            x, e1_e2_start, labels, _, _, _, _ = data
            attention_mask = (x != pad_id).float()
            token_type_ids = torch.zeros((x.shape[0], x.shape[1])).long()
            ## If GPU exist, allocate batch data into the GPU
            if cuda:
                x = x.cuda()
                labels = labels.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()

            classification_logits = net(
                x,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                Q=None,
                e1_e2_start=e1_e2_start,
            )
            # Calculate the cost function for the current batch
            loss = criterion(classification_logits, labels.squeeze(1))
            loss = loss / args["gradient_acc_steps"]
            if args["fp16"]:
                # Scale loss for
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args["fp16"]:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args["max_norm"]
                )
            else:
                grad_norm = clip_grad_norm_(net.parameters(), args["max_norm"])

            if (i % args["gradient_acc_steps"]) == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            total_acc += evaluate_(classification_logits, labels, ignore_idx=-1)[0]

            if (i % update_size) == (update_size - 1):
                losses_per_batch.append(
                    args["gradient_acc_steps"] * total_loss / update_size
                )
                accuracy_per_batch.append(total_acc / update_size)
                logger.info(
                    "[Epoch: %d, %5d/ %d points] total loss, accuracy per batch: %.3f, %.3f"
                    % (
                        epoch + 1,
                        (i + 1) * args["batch_size"],
                        train_len,
                        losses_per_batch[-1],
                        accuracy_per_batch[-1],
                    )
                )
                total_loss = 0.0
                total_acc = 0.0

        scheduler.step()
        cr, roc_plot = evaluate_results(
            net, valid_loader, label_binarizer, inferer.id2label, pad_id, cuda
        )
        results = {
            "valid_accuracy": round(cr.loc["accuracy"]["f1-score"], 4),
            "valid_f1_macro": round(cr.loc["macro avg"]["f1-score"],4),
            "valid_recall": round(cr.loc["macro avg"]["recall"], 4),
            "valid_precision": round(cr.loc["macro avg"]["precision"], 4),
            **{
                f"{k}_f1_score": round(cr.loc[str(k)]["f1-score"], 4)
                for k in net.config.label2id.keys()
            },
        }
        epoch_logs = dict()
        epoch_logs["train_losses"] = sum(losses_per_batch) / len(losses_per_batch)
        epoch_logs["train_accuracy"] = sum(accuracy_per_batch) / len(accuracy_per_batch)
        epoch_logs = epoch_logs | results

        all_logs.append(epoch_logs)

        logger.info("Epoch finished, took %.2f seconds." % (time.time() - start_time))
        logger.info(
            "Losses at Epoch %d: %.7f" % (epoch + 1, epoch_logs["train_losses"])
        )
        logger.info(
            "Train accuracy at Epoch %d: %.7f"
            % (epoch + 1, epoch_logs["train_accuracy"])
        )
        logger.info(f"Valid results Epoch {epoch + 1}: {results}")

        if epoch_logs["valid_f1_macro"] > best_pred:
            logger.info(
                f'Saving new checkpoint, the model score improved from {best_pred} to {epoch_logs["valid_f1_macro"]}'
            )

            best_pred = epoch_logs["valid_f1_macro"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "best_score": best_pred,
                    "scheduler": scheduler.state_dict(),
                    "amp": amp.state_dict() if amp is not None else amp,
                },
                model_artifacts,
            )

            net.config.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            j_resutls = open(src_dir/"metrics/metrics.json", "w+")
            json.dump(results, j_resutls)
            j_resutls.close()
            roc_plot.savefig(src_dir/"metrics/valid_roc_auc.png")
        elif args.get("overwrite"):
            logger.info(
                f'Saving new checkpoint, the model score improved from {best_pred} to {epoch_logs["valid_f1_macro"]}'
            )

            best_pred = epoch_logs["valid_f1_macro"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "best_score": best_pred,
                    "scheduler": scheduler.state_dict(),
                    "amp": amp.state_dict() if amp is not None else amp,
                },
                model_artifacts,
            )

            net.config.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            j_resutls = open(src_dir/"metrics/metrics.json", "w+")
            json.dump(results, j_resutls)
            j_resutls.close()
            roc_plot.savefig(src_dir/"metrics/valid_roc_auc.png")
        
        if args.get('reload') and epoch < args["num_epochs"] - 1:
            shuffle = not shuffle
            print("shuffle", shuffle)
            (
                train_loader,
                valid_loader,
                train_len,
                valid_len,
                rm,
                label_binarizer,
                weights
            ) = load_dataloaders(args, shuffle=shuffle)


    logs = pd.DataFrame(all_logs)
    logs["epoch"] = logs.index
    for col in logs.columns.drop("epoch").tolist():
        logs[[col, "epoch"]].to_csv(src_dir/f"metrics/{col}.csv", index=False)

    return net
