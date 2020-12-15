import torch
import os
import csv
from tqdm import tqdm, trange
import time


def evaluate(model, tokenizer, device, test_dataloader):
    """Une simple fonction permettant d'évaluer la précision du modèle sur
    un jeu de test"""

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    total = 0
    correct = 0
    sum_loss = 0

    for batch_id, batch in enumerate(test_dataloader) :
        images, questions, answers = batch

        images = images.to(device)
        answers = answers.to(device)

        encoding = tokenizer(
            list(questions),
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # On fait la prédiction sur le batch de test.
        pred = model(images, tokens)

        loss = loss_fn(pred, answers)
        sum_loss += loss.item()

        pred = torch.nn.Softmax(dim = 1)(pred)
        _, pred = torch.max(pred, 1)

        total += answers.size(0)
        correct += (pred == answers).sum().item()

    accuracy = 100 * correct / total
    return sum_loss, accuracy


def train_optim(
    model,
    tokenizer,
    train_dataloader,
    test_dataloader,
    loss_fn,
    epochs,
    log_frequency,
    device,
    save_file="",
    save_dir="",
    learning_rate=1e-4):

    # Suppression du fichier de sauvegarde si il existe déja.
    if save_file and os.path.isfile(save_file):
        os.remove(save_file)
    
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training in progress...")

    start_time = time.time()
    for t in (r := trange(epochs)) :
        model.train()
        sum_loss = 0

        for batch_id, batch in enumerate(train_dataloader):
            images, questions, answers = batch
            images = images.to(device)
            answers = answers.to(device)

            encoding = tokenizer(list(questions),
                                truncation=True,
                                padding=True,
                                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            tokens = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            pred = model(images, tokens)

            loss = loss_fn(pred, answers)
            sum_loss += loss.item()

            optimizer.zero_grad() # clear the gradient before backward
            loss.backward()       # update the gradient
            optimizer.step()      # update the model parameters using the gradient
        
        model.eval()
        test_loss, accuracy = evaluate(model, tokenizer, device, test_dataloader)
        
        if (t + 1) % log_frequency == 0 or t == 0:
            tqdm.write("Epoch: {:03d}, Training loss: {:.3f}, Test loss: {:.3f}, Evaluation accuracy: {:.3f} ".format(t + 1, sum_loss, test_loss, accuracy))
            if save_dir: torch.save(model, f"{save_dir}/model_{t + 1}.pt")

        if save_file: save_info(save_file, t, sum_loss, test_loss, accuracy, time.time() - start_time)
        r.set_description(f"epoch: {t + 1}, training loss: {sum_loss:.3f}, test loss: {test_loss:.3f}, accuracy: {accuracy:.3f}")

    print(f"--- {(time.time() - start_time):.2f} seconds ---")

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def save_info(file_path, epoch, loss, test_loss, accuracy, duration):
    fieldnames = ["epoch", "loss", "test_loss", "accuracy", "duration"]
    if os.path.isfile(file_path) == False:
        with open(file_path, "w") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
    with open(file_path, "a") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({
            "epoch": epoch,
            "loss": loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "duration": duration
        })