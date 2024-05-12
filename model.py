import torch
from torch import nn
from torchinfo import summary
from torchvision import models
import torch.nn.functional as F
from torchvision.models import VGG16_Weights


last_person_id = 16

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation layer

    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Squeeze
        self.fc = nn.Sequential(  # Excitation (similar to attention)
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FinalModel(nn.Module):
    def __init__(self):
        super(FinalModel, self).__init__()

        self.subject_biases = nn.Parameter(torch.zeros(last_person_id * 2, 2))  # pitch and yaw offset for the original and mirrored participant
        vgg_weights = VGG16_Weights.DEFAULT
        self.cnn_face = nn.Sequential(
            models.vgg16(weights=vgg_weights).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(11, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.cnn_eye = nn.Sequential(
            models.vgg16(weights=vgg_weights).features[:9],  # first four convolutional layers of VGG16 pretrained on ImageNet
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(2, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(4, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding='valid', dilation=(5, 11)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )

        self.fc_face = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6 * 6 * 128, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
        )

        self.cnn_eye2fc = nn.Sequential(
            SELayer(256),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            SELayer(256),

            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            SELayer(128),
        )

        self.fc_eye = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 6 * 128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc_eyes_face = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(576, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
        )

        self.validation_outputs = []

    def forward(self, person_idx: torch.Tensor, full_face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
        out_cnn_face = self.cnn_face(full_face)
        out_fc_face = self.fc_face(out_cnn_face)

        out_cnn_right_eye = self.cnn_eye(right_eye)
        out_cnn_left_eye = self.cnn_eye(left_eye)
        out_cnn_eye = torch.cat((out_cnn_right_eye, out_cnn_left_eye), dim=1)

        cnn_eye2fc_out = self.cnn_eye2fc(out_cnn_eye)  # feature fusion
        out_fc_eye = self.fc_eye(cnn_eye2fc_out)

        fc_concatenated = torch.cat((out_fc_face, out_fc_eye), dim=1)
        t_hat = self.fc_eyes_face(fc_concatenated)  # subject-independent term

        return t_hat + self.subject_biases[person_idx].squeeze(1)  # t_hat + subject-dependent bias term
    
    # Assuming you aggregate test outputs in a similar manner to validation outputs
    def on_test_epoch_end(self) -> None:
        # Perform operations similar to on_validation_epoch_end
        # Example: calculate and log the average test loss
        if hasattr(self, 'test_outputs') and len(self.test_outputs) > 0:
            avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
            self.log('avg_test_loss', avg_loss)
            # Reset for the next test run
            self.test_outputs = []
        else:
            print("No test outputs to process.")
    def _loss_function(self, outputs, labels):
        return F.mse_loss(outputs, labels)
    
    def calculate_loss_and_labels(self, batch: dict):
        person_idx = batch['person_idx']
        full_face = batch['full_face_image']
        right_eye = batch['right_eye_image']
        left_eye = batch['left_eye_image']
        labels = torch.stack([batch['gaze_pitch'], batch['gaze_yaw']], dim=1)
        
        # Assuming you have a method to calculate loss already defined
        outputs = self.forward(person_idx, full_face, right_eye, left_eye)
        loss = self._loss_function(outputs, labels)

        return loss, labels, outputs
    
    def on_validation_epoch_end(self) -> None:
        """# Code to run at the end of the validation epoch
        # Here you might want to log aggregated metrics or perform
        # some sort of validation epoch end processing
        
        # Example: Log the average validation loss and reset tracking variable
        avg_loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        # If you've stored the validation outputs during the epoch
        # you might want to do something with them now
        # Don't forget to reset the validation_outputs for the next epoch
        self.validation_outputs = []
        
        # Example: Log a figure using validation data
        # You will replace this with your own logic
        # figure = your_plotting_function(...)
        # self.logger.experiment.add_figure('validation/figure', figure, self.current_epoch)
        """

        if hasattr(self, 'validation_outputs') and len(self.validation_outputs) > 0:
            # We assume each 'loss' is a 0-dim tensor; if it's not, `.mean()` will give an error
            avg_loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean()
            self.log('avg_val_loss', avg_loss)
        else:
            print("No validation outputs to log.")

        # Reset the validation outputs for the next epoch
        self.validation_outputs = []

if __name__ == '__main__':

    model = FinalModel()

    #model.summarize(max_depth=1)
    #summary(model, input_size=(batch_size, 3, 224, 224))

    print(model.cnn_face)

    batch_size = 16
    summary(model, [
        (batch_size, 1),
        (batch_size, 3, 96, 96),  # full face
        (batch_size, 3, 64, 96),  # right eye
        (batch_size, 3, 64, 96)  # left eye
    ], dtypes=[torch.long, torch.float, torch.float, torch.float])
