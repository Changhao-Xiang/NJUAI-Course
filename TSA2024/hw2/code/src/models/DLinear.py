import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from src.models.base import MLForecastModel
from src.utils.decomposition import differential_decomposition, moving_average
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


class DLinear(MLForecastModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = Model(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.channels = args.enc_in

    def _fit(self, X: np.ndarray, epochs: int = 10, batch_size: int = 32) -> None:
        self.model.train()

        # Convert ndarray data to tensors
        X_s = sliding_window_view(X[0, :, :], (self.seq_len + self.pred_len, self.channels))
        X_s = X_s.reshape(-1, self.seq_len + self.pred_len, self.channels)
        train_X = X_s[:, : self.seq_len, :]
        train_Y = X_s[:, self.seq_len :, :]

        X_tensor = torch.from_numpy(train_X).float().to(self.device)
        y_tensor = torch.from_numpy(train_Y).float().to(self.device)

        # Build dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

    def _forecast(self, X: np.ndarray, pred_len) -> np.ndarray:
        self.model.eval()

        # Convert input numpy array to PyTorch tensor
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            forecast = self.model(X_tensor)

        # Convert the output tensor to numpy array
        return forecast.cpu().numpy()


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        # self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = individual
        self.channels = configs.enc_in

        self.decomposition = configs.decomposition

        if individual:
            # Separate weights for each variate
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)]
            )
        else:
            # Shared weights for all variates
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # Decompose input into trend and seasonal components
        if "ma" in self.decomposition:
            trend, seasonal = moving_average(np.array(x.cpu()), seasonal_period=24)
        elif "differential" in self.decomposition:
            trend, seasonal = differential_decomposition(np.array(x.cpu()))
        elif "no" in self.decomposition:
            raise ValueError("decomposition is required for DLinear model")

        trend = torch.from_numpy(trend).float().to(x.device)
        seasonal = torch.from_numpy(seasonal).float().to(x.device)

        if self.individual:
            # Apply individual linear layers for each variate
            trend_output = torch.stack(
                [self.linear_trend[i](trend[:, :, c]) for c in range(self.channels)], dim=-1
            )
            seasonal_output = torch.stack(
                [self.linear_seasonal[i](seasonal[:, :, c]) for c in range(self.channels)], dim=-1
            )
        else:
            # Apply shared linear layers
            trend_output = torch.stack(
                [self.linear_trend(trend[:, :, c]) for c in range(self.channels)], dim=-1
            )
            seasonal_output = torch.stack(
                [self.linear_seasonal(seasonal[:, :, c]) for c in range(self.channels)], dim=-1
            )

        # Combine trend and seasonal outputs
        forecast = trend_output + seasonal_output
        return forecast
