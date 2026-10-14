import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TelemetryGenerator:
    """
    Cam Ambalaj Hattı Sentetik Telemetri Veri Jeneratörü.
    
    Sistemin nominal (sıfır aşınma) verisini üretir ve istenen ekipmana
    durum bağımlı üstel aşınma / anomali sinyali enjekte eder.
    """

    def __init__(self, config_path: str = "config/line_config.yaml"):
        """
        Yapılandırma dosyasını yükler ve temel nominal değerleri ilklendirir.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {config_path}")
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.equipments = self.config.get("equipments", [])
        
        # Ekipman türlerine göre nominal varsayılan değer aralıkları
        self._equipment_baselines = {
            "IS_DRIVE": {"current": 12.0, "vibration": 0.8, "temp": 45.0, "pressure": 6.0},
            "CONV": {"current": 8.5, "vibration": 1.1, "temp": 42.0, "pressure": 0.0},
            "LEHR": {"current": 25.0, "vibration": 1.5, "temp": 85.0, "pressure": 2.5},
            "PUMP": {"current": 6.0, "vibration": 0.9, "temp": 38.0, "pressure": 4.5},
            "COMPRESSOR": {"current": 45.0, "vibration": 2.2, "temp": 70.0, "pressure": 8.0},
            "DEFAULT": {"current": 10.0, "vibration": 1.0, "temp": 40.0, "pressure": 5.0}
        }

    def _get_base_values(self, equip_id: str) -> Dict[str, float]:
        """Ekipman ID'sine göre nominal referans değerlerini döndürür."""
        for key, vals in self._equipment_baselines.items():
            if key in equip_id:
                return vals
        return self._equipment_baselines["DEFAULT"]

    def generate_golden_baseline(
        self, 
        hours: float = 250.0, 
        sampling_rate_sec: int = 10,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        İlk devreye alma (Golden Baseline - 0-250 saat) nominal sağlıklı veri setini üretir.
        
        Args:
            hours (float): Üretilecek simülasyon süresi (saat).
            sampling_rate_sec (int): Veri örnekleme periyodu (saniye).
            random_seed (int): Tekrarlanabilirlik için rastgele sayı üreteci tohumu.
            
        Returns:
            pd.DataFrame: Tüm ekipmanların nominal telemetri zaman serisi.
        """
        np.random.seed(random_seed)
        total_steps = int((hours * 3600) / sampling_rate_sec)
        timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=total_steps, freq=f"{sampling_rate_sec}s")
        
        records = []

        # Genel hat çalışma yükü profili (0.8 - 1.0 arası hafif dalgalanan çalışma)
        base_load = 0.9 + 0.05 * np.sin(np.linspace(0, 10 * np.pi, total_steps))

        for eq in self.equipments:
            eq_id = eq["id"]
            base = self._get_base_values(eq_id)
            
            # Rastgele proses gürültüleri (Gauss gürültüsü)
            current_noise = np.random.normal(0, 0.15, total_steps)
            vib_noise = np.random.normal(0, 0.03, total_steps)
            temp_noise = np.random.normal(0, 0.5, total_steps)
            press_noise = np.random.normal(0, 0.05, total_steps)

            # Nominal fiziksel tepkiler (Girdi yükü ile orantılı)
            current_rms = base["current"] * base_load + current_noise
            vibration_rms = base["vibration"] * (0.8 + 0.2 * base_load) + vib_noise
            temperature = base["temp"] + 5.0 * (base_load - 0.9) + temp_noise
            pressure = base["pressure"] + press_noise if base["pressure"] > 0 else np.zeros(total_steps)

            # Negatif değerleri engelle
            current_rms = np.maximum(0.1, current_rms)
            vibration_rms = np.maximum(0.01, vibration_rms)
            temperature = np.maximum(15.0, temperature)
            pressure = np.maximum(0.0, pressure)

            eq_df = pd.DataFrame({
                "timestamp": timestamps,
                "equipment_id": eq_id,
                "load_ratio": base_load,
                "current_rms": current_rms,
                "vibration_rms": vibration_rms,
                "temperature": temperature,
                "pressure": pressure,
                "is_degraded": False
            })
            records.append(eq_df)

        df_all = pd.concat(records, ignore_index=True)
        return df_all

    def inject_degradation(
        self,
        df: pd.DataFrame,
        target_equipment_id: str,
        start_hour: float = 100.0,
        degradation_rate: float = 0.05,
        exponent: float = 2.0
    ) -> pd.DataFrame:
        """
        Belirtilen ekipmana zaman ve yüke bağlı üstel aşınma/anomali enjekte eder.
        
        dD/dt = alpha * (load)^gamma * (D + eps)^kappa denkleminin basitleştirilmiş yansımasıdır.
        
        Args:
            df (pd.DataFrame): Ham telemetri DataFrame'i.
            target_equipment_id (str): Yıpratılacak ekipman ID'si (Örn: "IS_DRIVE_04").
            start_hour (float): Aşınmanın başlayacağı simülasyon saati.
            degradation_rate (float): Aşınma şiddeti katsayısı.
            exponent (float): Üstel katlanma derecesi (kappa).
            
        Returns:
            pd.DataFrame: Aşınma enjekte edilmiş telemetri.
        """
        df_out = df.copy()
        mask_eq = df_out["equipment_id"] == target_equipment_id
        
        if not mask_eq.any():
            raise ValueError(f"Hedef ekipman bulunamadı: {target_equipment_id}")

        time_series = df_out.loc[mask_eq, "timestamp"]
        start_time = time_series.iloc[0] + pd.Timedelta(hours=start_hour)
        
        mask_degrade = mask_eq & (df_out["timestamp"] >= start_time)
        
        num_degraded_steps = mask_degrade.sum()
        if num_degraded_steps == 0:
            return df_out

        # Geçen bağıl zaman (0 ile 1 arasında normalize zaman adımı)
        t_norm = np.linspace(0, 1, num_degraded_steps)
        
        # Yüke bağlı üstel aşınma faktörü: D(t) = rate * (t_norm)^exponent
        load_factors = df_out.loc[mask_degrade, "load_ratio"].values
        degradation_factor = degradation_rate * (t_norm ** exponent) * load_factors

        # Fiziksel büyüklüklerde bozulma yansıması
        # 1. Akım tepe değerinde artış (%20 - %60 artış)
        df_out.loc[mask_degrade, "current_rms"] += (
            df_out.loc[mask_degrade, "current_rms"] * degradation_factor * 1.5
        )
        
        # 2. Titreşimde katlanarak büyüme (Rulman/Dişli aşınması)
        df_out.loc[mask_degrade, "vibration_rms"] += (
            df_out.loc[mask_degrade, "vibration_rms"] * degradation_factor * 3.0
        )
        
        # 3. Sürtünmeden kaynaklı sıcaklık yükselmesi
        df_out.loc[mask_degrade, "temperature"] += degradation_factor * 25.0
        
        # 4. Aşınma bayrağını güncelle
        df_out.loc[mask_degrade, "is_degraded"] = True

        return df_out


if __name__ == "__main__":
    # Test çalıştırması ve doğrulama
    generator = TelemetryGenerator(config_path="config/line_config.yaml")
    
    print("Golden Baseline üretiliyor (10 saatlik hızlı test)...")
    df_baseline = generator.generate_golden_baseline(hours=10.0, sampling_rate_sec=10)
    print(f"Üretilen Satır Sayısı: {len(df_baseline)}")
    print(f"Kapsanan Ekipman Sayısı: {df_baseline['equipment_id'].nunique()}")
    
    print("\n'IS_DRIVE_04' ekipmanına 5. saatten itibaren aşınma enjekte ediliyor...")
    df_degraded = generator.inject_degradation(
        df_baseline, 
        target_equipment_id="IS_DRIVE_04", 
        start_hour=5.0, 
        degradation_rate=0.1
    )
    
    # Aşınma öncesi ve sonrası ortalama değer kıyaslaması
    eq_data = df_degraded[df_degraded["equipment_id"] == "IS_DRIVE_04"]
    normal_stats = eq_data[~eq_data["is_degraded"]][["current_rms", "vibration_rms", "temperature"]].mean()
    degraded_stats = eq_data[eq_data["is_degraded"]][["current_rms", "vibration_rms", "temperature"]].mean()
    
    print("\n--- IS_DRIVE_04 Nominal vs Aşınmış Değer Kıyaslaması ---")
    print(f"Nominal  -> Akım: {normal_stats['current_rms']:.2f}A, Titreşim: {normal_stats['vibration_rms']:.2f}mm/s, Sıcaklık: {normal_stats['temperature']:.2f}°C")
    print(f"Aşınmış  -> Akım: {degraded_stats['current_rms']:.2f}A, Titreşim: {degraded_stats['vibration_rms']:.2f}mm/s, Sıcaklık: {degraded_stats['temperature']:.2f}°C")