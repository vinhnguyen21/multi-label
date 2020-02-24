import os
import sys
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
import torch
import torch.utils.data as utils
from src.loss import lovasz_softmax, binary_xloss, FocalLoss
from src.dataset import RetinalDataset
from src.metric import multi_label_f1
from src.train_test_split import train_test_split, prepare_df
from src.transformation import inference_transformation, train_transformation
#loading model
from src.model import get_torchvision_model
##################
class Trainer(object):
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])

class RetinalDiseaseDetection(Trainer):
    def __init__(self, **args):
        super(RetinalDiseaseDetection, self).__init__(**args)
    def epoch_training(self, epoch, model,loss_criteria, device, train_loader, optimizer):
        model.train()
        training_loss = 0

        for batch, (inputs, labels) in enumerate(train_loader):
            inputs = torch.autograd.Variable(inputs).to(device, dtype=torch.float)
            labels = torch.autograd.Variable(labels).to(device, dtype=torch.float)

            optimizer.zero_grad()
            ##########feedforward############
            result = model(inputs)
            if self.loss.startswith("focal") or self.loss.startswith("smooth"):
                result = torch.sigmoid(result)
            ## Inception-V3
            if type(result) == tuple:
                outputs, aux_outputs = result
                loss1 = loss_criteria(outputs, labels)
                loss2 = loss_criteria(aux_outputs, labels)
                loss = (loss1 + 0.4*loss2)
            else:
                loss = loss_criteria(result, labels)
             #backward
#             loss.backward()
            loss.backward()
            optimizer.step()

#             scheduler.step()
            # Update training loss after each batch
            training_loss += loss.item()
            sys.stdout.write(f"\rEpoch {epoch+1}... Training step {batch+1}/{len(train_loader)}")
        # Clear memory after training
        print("Training Loss: "+ str(training_loss/len(train_loader)))
        del inputs, labels, loss
        # return training loss
        return training_loss/len(train_loader)

    def scoring(self, gt, pred, metric_name = 'F1'):
        """ override function
        """
        mean_score = 0
        class_scores = []
        if metric_name == 'AUROC':
            # Calculate AUROC
            class_scores = multi_label_auroc(gt, pred)
            mean_score = np.array(class_scores).mean()
        elif metric_name == 'F1':
            # Calculate F1
            class_scores = multi_label_f1(gt, pred)
            gt_np = gt.to("cpu").numpy()
            pred_np = (pred.to("cpu").numpy() > 0.5) * 1
            mean_score = f1_score(gt_np, pred_np, average='samples')
        # Log Score information
        print('---------------------------------------')
        print('F1 mean:{}'.format(mean_score))
        maxlen = len(max(self.classes, key=len))
        for i in range (0, len(class_scores)):
            score_val = '{:.4f}'.format(class_scores[i])
            print("{}\t{}".format(self.classes[i].ljust(maxlen, " "), score_val))
        sys.stdout.write('---------------------------------------')
        return mean_score

    def epoch_evaluating(self, model, loss_criteria, device, val_loader):
        model.eval()
        # Switch model to evaluation mode                               # Loss of model on validation set
        out_pred = torch.FloatTensor().to(device)      # Tensor stores prediction values
        out_gt = torch.FloatTensor().to(device)        # Tensor stores groundtruth values

        with torch.no_grad(): # Turn off gradient
            # For each batch
            for step, (images, labels) in enumerate(val_loader):
                # Transform X, Y to autogradient variables and move to device (GPU)
                images = images.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                # Update groundtruth values
                out_gt = torch.cat((out_gt, labels), 0)
                outputs = model(images)
                if type(outputs) == tuple:
                      outputs, aux_outputs = outputs
               # outputs, loss = self.feedforward(model, loss_criteria, images, labels)
                # Update prediction values
                out_pred = torch.cat((out_pred, outputs), 0)
             # Clear memory
        del images, labels
        # return validation loss, and metric score
        return self.scoring(out_gt, out_pred)

    def get_training_object(self):
        model = get_torchvision_model(self.net_type, self.pretrained, len(self.classes), self.loss)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if self.weight_path is not None:
            state_dict = torch.load(self.weight_path)
            # state_dict = state_dict["state"]
            model.load_state_dict(state_dict)
        #### optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr = self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "max", factor = self.factor, patience= self.patience)
        model = nn.DataParallel(model)
          #  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=10)

        if self.loss.startswith("lovasz"):
            loss_criteria = binary_xloss(ignore=255)
        elif self.loss.startswith("focal"):
            loss_criteria = FocalLoss(gamma=self.gamma)
        elif self.loss.startswith("smooth"):
            loss_criteria = nn.SmoothL1Loss()
        else:
            loss_criteria = nn.BCEWithLogitsLoss()
       # loss_criteria = nn.BCEWithLogitsLoss()
        return model, loss_criteria, optimizer, scheduler

    def save_model(self, model):
        os.makedirs(self.model_path, exist_ok = True)
        model_path = os.path.join(self.model_path, f'{self.net_type}_model.pth')
        saveModule = model
        saveModule = list(model.children())[0]
        model_specs = {"state": saveModule.state_dict(),
                       "grad_param_index": self.grad_param_index,
                       "grad_final_conv": self.grad_final_conv,
                       "net_type": self.net_type}
        torch.save(model_specs, model_path)
        return saveModule.state_dict()
    def train(self):
        best_score = 0

        # Init objects for training
        model, loss_criteria, optimizer, scheduler = self.get_training_object()

        #checking cuda
        if torch.cuda.is_available():
            print("training on GPU")
        else:
            print("training on CPU")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ##train_test_split
        print(self.dataframe)
        train =pd.read_csv(self.dataframe)
        X_train, y_train, X_test, y_test = train_test_split(train, self.target, self.classes)
        print(np.sum(y_train, axis=0))
        print(np.sum(y_test, axis=0))
        ##loading train_loader
        train_set = RetinalDataset(X_train, y_train, self.image_folder, self.size, train_transformation)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = self.batch_size, shuffle=True, num_workers=4)

        ## loading validation_loader
        valid_set = RetinalDataset(X_test, y_test, self.image_folder, self.size, inference_transformation)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = self.batch_size, shuffle=False, num_workers=4)

        ### training process
        for epoch in range(self.epoch):
            train_loss = self.epoch_training(epoch, model, loss_criteria, device, train_loader, optimizer)

            ## epoch evalulating
            new_score = self.epoch_evaluating(model, loss_criteria, device, valid_loader)
            scheduler.step(new_score)
            if best_score <= new_score:
                best_score = new_score
                state_dict = self.save_model(model)
#                 best_model_wts = copy.deepcopy(state_dict)
#         model.load_state_dict(best_model_wts)
        print("Best score: " + str(best_score))
        return model
