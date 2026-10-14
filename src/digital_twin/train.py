import sys
from pathlib import Path

# --- DİNAMİK KÖK DİZİN (PROJECT ROOT) BULUCU ---
def get_project_root() -> Path:
    """Proje ana klasörünü config/line_config.yaml dosyasını arayarak otomatik bulur."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config" / "line_config.yaml").exists() or (parent / "requirements.txt").exists():
            return parent
    return current.parent.parent

project_root = get_project_root()
src_path = project_root / "src"

# sys.path yollarına proje dizinlerini ekle
for path_to_add in [str(project_root), str(src_path)]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

# --- ESNEK IMPORT (src/ olan veya olmayan dizin yapıları için) ---
try:
    from src.data.generator import TelemetryGenerator
except ModuleNotFoundError:
    from data.generator import TelemetryGenerator

try:
    from src.digital_twin.model import LSTMRegressor
except ModuleNotFoundError:
    from digital_twin.model import LSTMRegressor

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from typing import Tuple, Dict


class TimeSeriesDataset(Dataset):
    """Zaman serisi kayan pencere (sliding window) PyTorch Dataset sınıfı."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 50):
        self.seq_len = seq_len
        self.X_data = []
        self.y_data = []

        for i in range(len(X) - seq_len):
            self.X_data.append(X[i : i + seq_len])
            self.y_data.append(y[i + seq_len])

        self.X_data = torch.tensor(np.array(self.X_data), dtype=torch.float32)
        self.y_data = torch.tensor(np.array(self.y_data), dtype=torch.float32)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]


class DigitalTwinTrainer:
    """Dijital İkiz Modellerinin Eğitim ve Saklama Yöneticisi."""

    def __init__(
        self, 
        config_path: str = "config/line_config.yaml",
        models_dir: str = "models_saved",
        seq_len: int = 50
    ):
        self.project_root = get_project_root()
        self.models_dir = self.project_root / models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len

        self.generator = TelemetryGenerator(config_path=str(self.project_root / config_path))

    def train_all_twins(self, epochs: int = 15, batch_size: int = 64, lr: float = 0.001):
        """Tüm ekipmanlar için Golden Baseline üzerinden ikizleri eğitir ve kaydeder."""
        print("1. Golden Baseline verisi üretiliyor (250 saatlik nominal çalışma)...")
        df_all = self.generator.generate_golden_baseline(hours=250.0, sampling_rate_sec=10)

        # Rodaj (Run-in) Filtreleme: İlk 48 saat kesilip atılır
        start_time = df_all["timestamp"].min() + pd.Timedelta(hours=48)
        df_train = df_all[df_all["timestamp"] >= start_time].copy()

        print(f"   Rodaj sonrası eğitim satır sayısı: {len(df_train)}")

        scalers: Dict[str, Dict[str, MinMaxScaler]] = {}
        unique_equipments = df_train["equipment_id"].unique()

        print(f"\n2. Toplam {len(unique_equipments)} ekipman için LSTM İkizleri eğitiliyor...\n")

        for eq_id in unique_equipments:
            eq_df = df_train[df_train["equipment_id"] == eq_id].sort_values("timestamp")

            # Girdi (u) ve Çıktı (y) Özellikleri
            X_raw = eq_df[["load_ratio"]].values
            y_raw = eq_df[["current_rms", "vibration_rms", "temperature"]].values

            # Min-Max Normalizasyonu
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_scaled = scaler_X.fit_transform(X_raw)
            y_scaled = scaler_y.fit_transform(y_raw)

            scalers[eq_id] = {"scaler_X": scaler_X, "scaler_y": scaler_y}

            # PyTorch Dataset & DataLoader
            dataset = TimeSeriesDataset(X_scaled, y_scaled, seq_len=self.seq_len)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Model, Kayıp ve Optimizatör
            model = LSTMRegressor(input_dim=1, hidden_dim=64, num_layers=2, output_dim=3, dropout=0.2)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Eğitim Döngüsü
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    preds = model(batch_X)
                    loss = criterion(preds, batch_y)
                    loss.backward()
                    optimizer.step()

            # Değerlendirme ($R^2$ Skoru)
            model.eval()
            with torch.no_grad():
                all_X, all_y = dataset.X_data, dataset.y_data
                all_preds = model(all_X).numpy()
                all_y_true = all_y.numpy()

            r2 = r2_score(all_y_true, all_preds, multioutput="variance_weighted")
            print(f" -> [{eq_id}] Eğitildi | Son Epoch MSE: {loss.item():.6f} | R² Skoru: {r2:.4f}")

            # Eğitilmiş Model Ağırlıklarını Dondurarak Kaydet
            model_path = self.models_dir / f"twin_{eq_id}.pt"
            torch.save(model.state_dict(), model_path)

        # Ölçekleyicileri (Scalers) Kaydet
        scalers_path = self.models_dir / "scalers.pkl"
        joblib.dump(scalers, scalers_path)
        print(f"\nTüm model ağırlıkları ve scalers.pkl '{self.models_dir}' altına dondurularak kaydedildi!")


if __name__ == "__main__":
    trainer = DigitalTwinTrainer()
    trainer.train_all_twins(epochs=12, batch_size=128, lr=0.002)