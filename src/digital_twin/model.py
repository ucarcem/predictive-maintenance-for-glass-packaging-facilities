import torch
import torch.nn as nn
from typing import Tuple


class LSTMRegressor(nn.Module):
    """
    Siyah Kutu Dijital İkiz için Dinamik LSTM Regresör Modeli.
    
    Girdi olarak zaman serisi dizisini (seq_len, input_dim) alır;
    çıktı olarak son zaman adımındaki nominal fiziki büyüklükleri
    (current_rms, vibration_rms, temperature vb.) tahmin eder.
    """

    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 64, 
        num_layers: int = 2, 
        output_dim: int = 3, 
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim (int): Girdi nitelik sayısı (Örn: load_ratio = 1).
            hidden_dim (int): LSTM gizli katman birim sayısı (default: 64).
            num_layers (int): Üst üste binen LSTM katmanı sayısı (default: 2).
            output_dim (int): Tahmin edilecek hedef sensör sayısı (default: 3 -> akım, titreşim, sıcaklık).
            dropout (float): Aşırı öğrenmeyi (overfitting) önleyici dropout oranı.
        """
        super(LSTMRegressor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Sıralı Çift Katmanlı LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout ve Tam Bağlantılı (Dense/Linear) Çıkış Katmanı
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass.
        
        Args:
            x (torch.Tensor): Tensor boyutu (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Tahmin edilen nominal değerler (batch_size, output_dim)
        """
        # lstm_out: (batch_size, seq_len, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Zaman dizisinin son adımındaki gizli durumu al (last time step)
        last_step_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        out = self.dropout(last_step_out)
        predictions = self.fc(out)  # (batch_size, output_dim)

        return predictions


if __name__ == "__main__":
    print("PyTorch LSTM Regresör Modeli Test Ediliyor...")
    
    # Örnek Hiperparametreler
    batch_size = 16
    seq_len = 50       # 50 zaman adımlı bakış penceresi
    input_dim = 1      # Girdi: load_ratio
    output_dim = 3     # Çıktılar: current_rms, vibration_rms, temperature

    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=output_dim,
        dropout=0.2
    )

    # Rastgele girdi tensörü oluştur (batch_size, seq_len, input_dim)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Model ileri besleme (Forward Pass)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Girdi Boyutu  : {dummy_input.shape} -> (batch_size={batch_size}, seq_len={seq_len}, input_dim={input_dim})")
    print(f"Model Çıktı Boyutu  : {output.shape}     -> (batch_size={batch_size}, output_dim={output_dim})")
    print(f"Model Parametre Sayısı: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("\nLSTM Model Sınıfı Başarıyla Test Edildi!")