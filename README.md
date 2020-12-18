# Utilisation :

## Contexte :
Ce programme est un script python permettant d'entrainer le modèle `MegaModel` décrit dans le fichier `models.py`.
Ce modèle permet de répondre "oui" ou "non" à une question fermée posée sur une image.

## Commandes :

### Liste des options

Pour obtenir la liste de toutes les options utilisables,

```bash
$ python training.py --help
usage: training.py [-h] [-v] [--batch_size BATCH_SIZE] [--freeze] [--epochs EPOCHS] [--log_frequency LOG_FREQUENCY]
                   [--learning_rate LEARNING_RATE] [--save_info SAVE_INFO] [--save_model SAVE_MODEL] [--hidden HIDDEN]
                   [--from_pretrained PRETRAINED_MODEL]
                   PATH IMAGE_FOLDER DESCRIPTOR

Entraine un model pour répondre à des questions en fonctions des informations contenues dans une image.

positional arguments:
  PATH
  IMAGE_FOLDER
  DESCRIPTOR

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --batch_size BATCH_SIZE, -b BATCH_SIZE
  --freeze, -f
  --epochs EPOCHS, -e EPOCHS
  --log_frequency LOG_FREQUENCY, -log LOG_FREQUENCY
  --learning_rate LEARNING_RATE, -r LEARNING_RATE
  --save_info SAVE_INFO, -si SAVE_INFO
  --save_model SAVE_MODEL, -sm SAVE_MODEL
  --hidden HIDDEN, -hd HIDDEN
  --from_pretrained PRETRAINED_MODEL, -p PRETRAINED_MODEL
```

### Lancer l'entraînement

La commande de base pour lancer l'entraînement avec les paramètres par défault est,

```bash
$ python training.py <Chemin du dossier contenant les données> <Répertoire des images> <Fichier .csv avec les questions>
```

Exemple :
```bash
$ python training.py ./boolean_answers_dataset_10000 boolean_answers_dataset_images_10000 boolean_answers_dataset_10000.csv
```

### Entraînement avec des paramètres personnalisé

Pour changer les paramètres utilisés par défault il suffit d'ajouter le flag de l'option puis sa nouvelle valeur.

Exemples :

Pour changer le nombre d'epoch à 500, 
```bash
$ python training.py ./boolean_answers_dataset_10000 boolean_answers_dataset_images_10000 boolean_answers_dataset_10000.csv -e 500
```

Par défault l'entraînement est en fine-tuning. Pour le passer en feature extractor, il suffit d'utiliser le flag `-f`.

Exemple : 

```bash
python training.py ./boolean_answers_dataset_10000 boolean_answers_dataset_images_10000 boolean_answers_dataset_10000.csv -e 500  -f
```
