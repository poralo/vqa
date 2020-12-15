import torch

class ImageModel(torch.nn.Module):
    """Modèle permettant de traiter les images"""

    def __init__(self, cnn_model, num_out=1000):
        super(ImageModel, self).__init__()
        self.model = cnn_model

        # On change la tête de classification du model de base.
        self.model.fc = torch.nn.Linear(
            in_features=512,
            out_features=num_out,
            bias=True
        )

    def forward(self, x):
        return self.model(x)


class TextModel(torch.nn.Module):
    """Modèle permettant de traiter le texte"""

    def __init__(self, transformer_model, num_out=1000):
        super(TextModel, self).__init__()
        self.model = transformer_model

        # On change la tête de classification du model de base.
        self.model.classifier = torch.nn.Linear(
            in_features=768,
            out_features=num_out,
            bias=True
        )

    def forward(self, input_ids, attention_mask):
        x = self.model(input_ids, attention_mask=attention_mask).logits
        return x


class MegaModel(torch.nn.Module):
    """
    Modèle permettant de traiter des questions à propos d'images.
    Ici, les couches CNN et transformer sont aggrégés en faisait un produit valeur par valeur.
    """

    def __init__(self, image_model, text_model, num_hidden=1000):
        super(MegaModel, self).__init__()

        self.image_model = image_model
        self.text_model = text_model

        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, 500),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 2)
        )

    def forward(self, images, tokens):
        image = self.image_model(images)
        question = self.text_model(**tokens)

        # Produit terme à terme des valeurs en sorties des deux réseaux.
        x = image * question

        return self.classification_head(x)

class MegaModelAggregator(torch.nn.Module):
    """
    Modèle permettant de traiter des questions à propos d'images.
    Il est différent du MegaModel par la façon d'agreger les couches CNN et transformers, ici
    les valeurs des neuronnes sont concaténés pour ensuite passer par la tête de classification.
    """

    def __init__(self, image_model, text_model, num_hidden=2000):
        super(MegaModelAggregator, self).__init__()

        self.image_model = image_model
        self.text_model = text_model

        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, 500),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 200),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 2)
        )

    def forward(self, images, tokens):
        image = self.image_model(images)
        question = self.text_model(**tokens)

        # Aggregation des valeurs en sortie des couches CNN et transformer.
        x = torch.cat((image, question), 1)

        return self.classification_head(x)