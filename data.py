import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from PIL import Image
import pandas as pd
from typing import Any, Callable, Optional, Tuple
import numpy as np


class VQADataset(Dataset):
  """
    This class loads a shrinked version of the VQA dataset (https://visualqa.org/)
    Our shrinked version focus on yes/no questions. 
    To load the dataset, we pass a descriptor csv file. 
    
    Each entry of the csv file has this form:

    question_id ; question_type ; image_name ; question ; answer ; image_id

  """
  def __init__(self, path : str, dataset_descriptor : str, image_folder : str, transform : Callable) -> None:
    """
      :param: path : a string that indicates the path to the image and question dataset.
      :param: dataset_descriptor : a string to the csv file name that stores the question ; answer and image name
      :param: image_folder : a string that indicates the name of the folder that contains the images
      :param: transform : a torchvision.transforms wrapper to transform the images into tensors 
    """
    super(VQADataset, self).__init__()
    self.descriptor = pd.read_csv(path + '/' + dataset_descriptor, delimiter=';')
    self.path = path 
    self.image_folder = image_folder
    self.transform = transform
    self.size = len(self.descriptor)
  
  def __len__(self) -> int:
    return self.size

  def __getitem__(self, idx : int) -> Tuple[Any, Any, Any]:
    """
      returns a tuple : (image, question, answer)
      image is a Tensor representation of the image
      question and answer are strings
    """
    image_name = self.path + '/' + self.image_folder + '/' + self.descriptor["image_name"][idx]

    image = Image.open(image_name).convert("RGB")

    # if image.mode != 'RGB' :
    #   image = transforms.Grayscale(num_output_channels=3)(image)

    image = self.transform(image)

    question = self.descriptor["question"][idx]

    answer = 0 if self.descriptor["answer"][idx] == 'no' else 1
    

    return (image, question, answer)

def load_dataloaders(path, image_folder, descriptor, batch_size=10):
    # Exemples de transformation.
    transform = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),     
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    vqa_dataset = VQADataset(path, descriptor, image_folder, transform=transform)

    # Permet de contrôler l'aléatoire pour pouvoir reproduire les résultats.
    np.random.seed(1)
    torch.manual_seed(1)

    indices = np.arange(0, len(vqa_dataset))
    np.random.shuffle(indices)
    train_prop = 0.75
    train_examples_size = int(len(indices) * 0.75)
    test_examples_size = - (len(indices) - train_examples_size)

    vqa_train_dataloader = DataLoader(
        vqa_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=SubsetRandomSampler(indices[:train_examples_size]))

    vqa_test_dataloader = DataLoader(
        vqa_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        sampler=SubsetRandomSampler(indices[test_examples_size:]))
    
    print(f"Train dataset size: {len(vqa_train_dataloader)}, test dataset size: {len(vqa_test_dataloader)}")

    return vqa_train_dataloader, vqa_test_dataloader