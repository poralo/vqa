import torch
from torchvision import models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import freeze_model, train_optim
from models import ImageModel, TextModel, MegaModel, MegaModelAggregator
from data import load_dataloaders
import argparse


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entraine un model pour répondre à des questions en fonctions des informations contenues dans une image."
    )

    parser.add_argument(
        "-v", "--version", action="version",
        version = f"{parser.prog} version 0.0.1"
    )

    parser.add_argument("PATH")
    parser.add_argument("IMAGE_FOLDER")
    parser.add_argument("DESCRIPTOR")
    parser.add_argument("--batch_size", "-b", type=int, default=10, dest="BATCH_SIZE")
    parser.add_argument("--freeze", "-f", default=False, dest="FREEZE", action='store_true')
    parser.add_argument("--epochs", "-e", type=int, default=100, dest="EPOCHS")
    parser.add_argument("--log_frequency", "-log", type=int, default=10, dest="LOG_FREQUENCY")
    parser.add_argument("--learning_rate", "-r", type=float, default=1e-4, dest="LEARNING_RATE")
    parser.add_argument("--save_info", "-si", type=str, default="", dest="SAVE_INFO")
    parser.add_argument("--save_model", "-sm", type=str, default="", dest="SAVE_MODEL")
    parser.add_argument("--hidden", "-hd", type=int, default=1000, dest="HIDDEN")
    parser.add_argument("--from_pretrained", "-p", type=str, default="", dest="PRETRAINED_MODEL")

    return parser

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training on {device}")

    cnn_model = models.resnet18(pretrained = True)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased"
    )

    if FREEZE:
        freeze_model(cnn_model)
        freeze_model(transformer_model)

    image_model = ImageModel(cnn_model, num_out=H)
    text_model = TextModel(transformer_model, num_out=H)

    if PRETRAINED_MODEL:
        model = torch.load(PRETRAINED_MODEL)
    else:
        model = MegaModel(image_model, text_model, num_hidden=H)
        # model = MegaModelAggregator(image_model, text_model, num_hidden=2*H)
    model.to(device)

    train_dataloader, test_dataloader = load_dataloaders(
        path=PATH,
        image_folder=IMAGE_FOLDER,
        descriptor=DESCRIPTOR,
        batch_size=BATCH_SIZE
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    train_optim(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        test_dataloader=train_dataloader,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        log_frequency=LOG_FREQUENCY,
        device=device,
        save_file=SAVE_INFO,
        save_dir=SAVE_MODEL,
        learning_rate=LEARNING_RATE 
    )

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    BATCH_SIZE = args.BATCH_SIZE
    DESCRIPTOR = args.DESCRIPTOR
    EPOCHS = args.EPOCHS
    FREEZE = args.FREEZE
    IMAGE_FOLDER = args.IMAGE_FOLDER
    LEARNING_RATE = args.LEARNING_RATE
    LOG_FREQUENCY = args.LOG_FREQUENCY
    PATH = args.PATH
    SAVE_INFO = args.SAVE_INFO
    SAVE_MODEL = args.SAVE_MODEL
    H = args.HIDDEN
    PRETRAINED_MODEL = args.PRETRAINED_MODEL

    main()