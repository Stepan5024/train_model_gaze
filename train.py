from argparse import ArgumentParser
import os
import pathlib
import sys
import time
from typing import Tuple
from torchinfo import summary

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from logger import create_logger
from mpii_face_gaze_dataset import get_dataloaders
from model import FinalModel
from utils import calc_angle_error

logger = None
model = None

class Model(FinalModel):
    def __init__(self, learning_rate: float = 0.001, weight_decay: float = 0., k=None, adjust_slope: bool = False, grid_calibration_samples: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.k = [9, 128] if k is None else k
        self.adjust_slope = adjust_slope
        self.grid_calibration_samples = grid_calibration_samples
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Set learning on {self.device} torch.cuda.is_available() {torch.cuda.is_available()}")
        self.to(self.device)
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


    def train_step(self, batch):
        logger.info("Training step started.")
        for k, v in batch.items():
            logger.info(f"Key: {k}, Type: {type(v)}")

        #batch = {k: v.to(device=self.model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        self.train()
        self.optimizer.zero_grad()
        logger.info("Optimizer gradients reset.")
        loss, labels, outputs = self.process_batch(batch)
        logger.info(f"Processed batch - Loss computed: {loss.item():.4f}")
        loss.backward()
        logger.info("Backpropagation complete.")
        self.optimizer.step()
        logger.info("Optimizer step completed.")
        angle_error = calc_angle_error(labels, outputs)
        logger.info(f"Angle error calculated: {angle_error:.4f}")
        return loss.item(), angle_error
    

    def eval_step(self, batch):
        logger.info("Evaluation step started.")
        self.eval()
        logger.info("Model set to evaluation mode.")
        with torch.no_grad():
            logger.info("Starting batch processing for evaluation...")
            loss, labels, outputs = self.process_batch(batch)
            logger.info(f"Batch processed - Loss computed: {loss.item():.4f}")
        angle_error = calc_angle_error(labels, outputs)
        logger.info(f"Angle error calculated: {angle_error:.4f}")
        
        return loss.item(), angle_error, labels, outputs
    

    def process_batch(self, batch):
        logger.info("Processing batch...")

        person_idx = batch['person_idx'].to(self.device).long()
        logger.info(f"Person indices: {person_idx}")
        left_eye_image = batch['left_eye_image'].to(self.device).float()
        right_eye_image = batch['right_eye_image'].to(self.device).float()
        full_face_image = batch['full_face_image'].to(self.device).float()
        #logger.info(f"Batch size for images: {left_eye_image.size(0)}")

        gaze_pitch = batch['gaze_pitch'].to(self.device).float()
        gaze_yaw = batch['gaze_yaw'].to(self.device).float()
        labels = torch.stack([gaze_pitch, gaze_yaw]).T
        #logger.info("Labels prepared.")

        outputs = self(person_idx, full_face_image, right_eye_image, left_eye_image)
        logger.info(f"Outputs model {outputs[:10]}")
        logger.info(f"Outputs generated: Shape {outputs.shape}")

        loss = F.mse_loss(outputs, labels)
        logger.info(f"Computed loss: {loss.item():.4f}")
        return loss, labels, outputs
    
    def save_model(self, path, epoch, is_best=False):
        logger.info(f"Saving model state at epoch {epoch}...")
        # Save in PyTorch and checkpoint formats
        pth_file_path = os.path.join(path, f"model_epoch_{epoch}.pth")
        torch.save(self.model.state_dict(), pth_file_path)
        logger.info(f"Model saved in PyTorch format at: {pth_file_path}")
        ckpt_file_path = os.path.join(path, f"model_epoch_{epoch}.ckpt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
        }, ckpt_file_path)
        logger.info(f"Checkpoint saved at: {ckpt_file_path}")

        # Prepare to save in ONNX format
        self.model.eval()
        logger.info("Model set to evaluation mode for ONNX export.")

        batch_size = 1
        channels = 3
    
    
        dummy_person_idx = torch.tensor([0], device=self.device)  # Assuming `person_idx` expects integer indexes.
        dummy_full_face = torch.randn(batch_size, channels, 96, 96, device=self.device)
        dummy_right_eye = torch.randn(batch_size, channels, 64, 96, device=self.device)
        dummy_left_eye = torch.randn(batch_size, channels, 64, 96, device=self.device)
        dummy_inputs = (dummy_person_idx, dummy_full_face, dummy_right_eye, dummy_left_eye)
        logger.info(f"Prepared dummy inputs for ONNX export with batch size {batch_size}.")

        # Create dummy inputs with batch size of 1
        """dummy_person_idx = torch.tensor([0])  # Single index for person_idx
        dummy_full_face = torch.randn(1, 3, 96, 96)  # 1x3x96x96 input for full face
        dummy_right_eye = torch.randn(1, 3, 64, 96)  # 1x3x64x96 input for right eye
        dummy_left_eye = torch.randn(1, 3, 64, 96)  # 1x3x64x96 input for left eye
        
        """
        # Export model to ONNX format
        onnx_file_path = os.path.join(path, f"model_epoch_{epoch}.onnx")
        torch.onnx.export(self.model, dummy_inputs, onnx_file_path, export_params=True)
        logger.info(f"Model exported in ONNX format at: {onnx_file_path}")
        if is_best:
            best_model_path = os.path.join(path, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
            logger.info(f"Best model saved separately at: {best_model_path}")

    """
    def save_model(self, path, epoch, is_best=False):
        # Save in PyTorch format
        torch.save(self.model.state_dict(), os.path.join(path, f"model_epoch_{epoch}.pth"))

        # Save in checkpoint format if there's specific state to maintain
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
        }, os.path.join(path, f"model_epoch_{epoch}.ckpt"))

        # Save in ONNX format (Ensure the model is in evaluation mode and dummy input is correct)
        self.model.eval()
        
        batch_size = 1
        channels = 3


        dummy_person_idx = torch.tensor([0])  # Assuming `person_idx` expects integer indexes.
        dummy_full_face = torch.randn(batch_size, channels, 96, 96)
        dummy_right_eye = torch.randn(batch_size, channels, 64, 96)
        dummy_left_eye = torch.randn(batch_size, channels, 64, 96)
        dummy_inputs = (dummy_person_idx, dummy_full_face, dummy_right_eye, dummy_left_eye)

        torch.onnx.export(self.model, dummy_inputs, os.path.join(path, f"model_epoch_{epoch}.onnx"), export_params=True)

        if is_best:
            torch.save(self.model.state_dict(), os.path.join(path, "best_model.pth"))
        """

    def train_model(self, train_loader, valid_loader, num_epochs=1):
        total_start_time = time.time()
        logger.info(f"Starting training for {num_epochs} epochs...")
        total_batches = len(train_loader) * num_epochs
        best_valid_loss = float('inf')
        best_epoch = 0

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_losses, train_errors = [], []
            valid_losses, valid_errors = [], []
            batch_count_train = 0
            logger.info(f"Epoch {epoch+1}/{num_epochs} training start.")
            batch_times = []
            

            for i, batch in enumerate(train_loader):
                logger.info(f"Processing batch {i+1} of {len(train_loader)}")
                batch_start_time = time.time()
                batch_count_train += 1
                loss, error = self.train_step(batch)
                train_losses.append(loss)
                train_errors.append(error)
                batch_duration = time.time() - batch_start_time
                batch_times.append(batch_duration)

                if batch_count_train % 10 == 0:  # Adjust this modulo check to control verbosity
                    logger.info(f"  Training batch {batch_count_train}: Loss = {loss:.4f}, Error = {error:.4f}")
                if len(batch_times) > 1:
                    average_batch_time = sum(batch_times) / len(batch_times)
                    remaining_batches = total_batches - (epoch * len(train_loader) + batch_count_train)
                    estimated_time_remaining = average_batch_time * remaining_batches
                    logger.info(f"Estimated time remaining: {estimated_time_remaining / 60:.2f} minutes")
            

            logger.info(f"Completed training {batch_count_train} batches.")

            batch_count_valid = 0
            for batch in valid_loader:
                batch_count_valid += 1
                loss, error, _, _ = self.eval_step(batch)
                valid_losses.append(loss)
                valid_errors.append(error)
                if batch_count_valid % 10 == 0:  # Adjust this modulo check to control verbosity
                    logger.info(f"  Validation batch {batch_count_valid}: Loss = {loss:.4f}, Error = {error:.4f}")
            logger.info(f"Completed validation {batch_count_valid} batches.")
            
            epoch_train_loss = np.mean(train_losses)
            epoch_train_error = np.mean(train_errors)
            epoch_valid_loss = np.mean(valid_losses)
            epoch_valid_error = np.mean(valid_errors)
            logger.info(f"Epoch {epoch+1} Summary:")
            logger.info(f"  Average Training Loss: {epoch_train_loss:.4f}")
            logger.info(f"  Average Training Error: {epoch_train_error:.4f}")
            logger.info(f"  Average Validation Loss: {epoch_valid_loss:.4f}")
            logger.info(f"  Average Validation Error: {epoch_valid_error:.4f}")
            logger.info("--------------------------------------------------")

            # Save the model after each epoch
            save_path = f"F:\\EyeGazeDataset\\gaze_user"
            
            self.save_model(save_path, epoch)
            # Update best model
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                best_epoch = epoch
                self.save_model(save_path, best_epoch, is_best=True)
            epoch_duration = time.time() - epoch_start_time
            average_loss = sum(train_losses) / len(train_losses)
            average_error = sum(train_errors) / len(train_errors)
            logger.info(f"End of epoch {epoch+1}: Average Loss = {average_loss:.4f}, Average Error = {average_error:.4f}, Epoch Duration = {epoch_duration:.2f} seconds")

        logger.info("Training completed.")



def main(path_to_data: str, validate_on_person: int, test_on_person: int, learning_rate: float, weight_decay: float, batch_size: int, k: int, adjust_slope: bool, grid_calibration_samples: bool):
    global model
    model = Model(learning_rate, weight_decay, k, adjust_slope, grid_calibration_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Set learning torch.cuda.is_available() {torch.cuda.is_available()}")
    batch_size_dummy = 1
    channels = 3

    dummy_person_idx = torch.tensor([0], device=device)  # Assuming `person_idx` expects integer indexes.
    dummy_full_face = torch.randn(batch_size_dummy, channels, 96, 96, device=device)
    dummy_right_eye = torch.randn(batch_size_dummy, channels, 64, 96, device=device)
    dummy_left_eye = torch.randn(batch_size_dummy, channels, 64, 96, device=device)
    
    summary(model, input_data=(dummy_person_idx, dummy_full_face, dummy_right_eye, dummy_left_eye))

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(path_to_data, validate_on_person, test_on_person, batch_size)
    model.train_model(train_dataloader, valid_dataloader, num_epochs=2)

def resource_path(relative_path) -> str:
        """Возвращает корректный путь для доступа к ресурсам после сборки .exe"""
        #if getattr(sys, 'frozen', False):
        try:
            # PyInstaller создаёт временную папку _MEIPASS для ресурсов
            base_path = sys._MEIPASS
        except Exception:
            # Если приложение запущено из исходного кода, то используется обычный путь
            base_path = os.path.abspath(".")
    
        return os.path.join(base_path, relative_path)

def initLogger():
    global logger
    rev_path =  os.path.join("logs", "training", "gaze")
    abs_path = pathlib.Path(resource_path(rev_path))
    logger = create_logger("TrainGaze", abs_path, 'train_gaze_log.txt')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--path_to_data", type=str, default=r'F:\EyeGazeDataset\MPIIFaceGaze_post_proccessed_stepa_pperle')
    parser.add_argument("--validate_on_person", type=int, default=1)
    parser.add_argument("--test_on_person", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=[9, 128], nargs='+')
    parser.add_argument("--adjust_slope", type=bool, default=False)
    parser.add_argument("--grid_calibration_samples", type=bool, default=False)
    args = parser.parse_args()
    initLogger()
    main(args.path_to_data, args.validate_on_person, args.test_on_person, args.learning_rate, args.weight_decay, args.batch_size, args.k, args.adjust_slope, args.grid_calibration_samples)