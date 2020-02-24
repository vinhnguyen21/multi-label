import os
import torch
import torchvision
from torchvision import transforms
import numpy
from PIL import Image
from .model import get_torchvision_model
from .transformation import inference_transformation
from torch.utils.model_zoo import load_url
url = "https://storage.googleapis.com/v-project/Inceptionresnet-55513bc1.pth"
class EyescanDiseaseClassifier():
    def __init__(self, model_type = "Inceptionresnet", weight_path = None):
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type:
            self.model = get_torchvision_model(model_type, True, 6, None)

        #checking weight of model
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location=self.device)
        else:
            state_dict = load_url(url, map_location=self.device)
            print("Download weight success")
        state_dict = state_dict["state"]
        self.model.load_state_dict(state_dict)
        self.classes = [
    "Macular irregularity, may be a sign of age-related macular degeneration",
    "Optic nerve irregularity",
    "Optic nerve irregularity, may be a sign of glaucoma",
    "Eyescan findings are unremarkable",
    "Signs of retinal thinning",
    "Others"
  ]
        self.size = 512
    def transform(self, image):
        return inference_transformation(image, self.size)

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        #result
        prob = {}

        # predict Eyescan findings
        self.model.eval()
        with torch.no_grad():
            ps = self.model(image)
            ps = ps.detach().numpy().tolist()
            for idx, disease in enumerate(self.classes):
                prob[disease] = float(ps[0][idx])
        return prob
