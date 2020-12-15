# Commands:

## Pas de fine-tuning : batch 10 : dot product : 100 epoch
```bash
python training.py ./boolean_answers_dataset_10000 boolean_answers_dataset_images_10000 boolean_answers_dataset_10000.csv -e 100 -si info_freeze_hidden_10000.csv -sm freeze_model_dotprod_10000 -log 10 -b 10 -f
```

## Fine-tuning : batch 10 : dot product : 100 epoch
```bash
python training.py ./boolean_answers_dataset_10000 boolean_answers_dataset_images_10000 boolean_answers_dataset_10000.csv -e 100 -si info_no_freeze_hidden_10000.csv -sm no_freeze_model_dotprod_10000 -log 10 -b 10
```