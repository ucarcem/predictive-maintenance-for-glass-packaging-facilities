import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List


class DatabaseHandler:
    """
    Cam Ambalaj Hattı Veri Tabanı Yönetim Katmanı (SQLite / TimescaleDB Uyumlu).
    
    Sensör telemetri verilerini, dijital ikiz kalıntılarını, sağlık indekslerini
    ve MILP optimizasyonu tarafından üretilen iş emirlerini yönetir.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Veri tabanı bağlantısını kurar ve tabloları otomatik oluşturur.
        """
        if db_path is None:
            # Proje kök dizininde data/ klasörü altında varsayılan veritabanı
            project_root = Path(__file__).resolve().parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = data_dir / "maintenance_system.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """SQLite veritabanı bağlantısı döndürür."""
        return sqlite3.connect(self.db_path)

    def _init_tables(self) -> None:
        """
        Gerekli tüm veritabanı şema tablolarını ilklendirir.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Ham Sensör Telemetri Tablosu
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                equipment_id TEXT NOT NULL,
                load_ratio REAL NOT NULL,
                current_rms REAL NOT NULL,
                vibration_rms REAL NOT NULL,
                temperature REAL NOT NULL,
                pressure REAL NOT NULL,
                is_degraded BOOLEAN NOT NULL
            )
            """)

            # 2. Dijital İkiz Kalıntı (Residual) ve Mahalanobis Mesafesi Tablosu
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS residuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                equipment_id TEXT NOT NULL,
                current_residual REAL NOT NULL,
                vibration_residual REAL NOT NULL,
                temperature_residual REAL NOT NULL,
                mahalanobis_distance REAL NOT NULL
            )
            """)

            # 3. Sağlık İndeksi (HI) ve Olasılıksal RUL Tahmin Tablosu
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_rul (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                equipment_id TEXT NOT NULL,
                health_index REAL NOT NULL,
                rul_days_mean REAL NOT NULL,
                rul_days_std REAL NOT NULL,
                p_failure_28d REAL NOT NULL
            )
            """)

            # 4. MILP Çıktısı İş Emirleri (Work Orders) Tablosu
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS work_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                day_index INTEGER NOT NULL,
                equipment_id TEXT NOT NULL,
                is_planned BOOLEAN NOT NULL,
                is_hot_section BOOLEAN NOT NULL,
                status TEXT DEFAULT 'PLANNED'
            )
            """)

            # İndeksler (Sorgu performansını artırmak için)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_telem_eq_time ON telemetry (equipment_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_res_eq_time ON residuals (equipment_id, timestamp)")
            conn.commit()

    def save_telemetry(self, df: pd.DataFrame) -> None:
        """Telemetri DataFrame'ini veritabanına kaydeder."""
        df_save = df.copy()
        if "timestamp" in df_save.columns:
            df_save["timestamp"] = pd.to_datetime(df_save["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            df_save.to_sql("telemetry", conn, if_exists="append", index=False)

    def load_telemetry(
        self, 
        equipment_id: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Telemetri verilerini sorgular ve DataFrame olarak döndürür.
        """
        query = "SELECT * FROM telemetry"
        params: List[Union[str, int]] = []

        if equipment_id:
            query += " WHERE equipment_id = ?"
            params.append(equipment_id)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

    def save_residuals(self, df: pd.DataFrame) -> None:
        """Kalıntı ve Mahalanobis mesafe sonuçlarını kaydeder."""
        df_save = df.copy()
        if "timestamp" in df_save.columns:
            df_save["timestamp"] = pd.to_datetime(df_save["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            df_save.to_sql("residuals", conn, if_exists="append", index=False)

    def save_work_orders(self, df: pd.DataFrame) -> None:
        """MILP optimizasyonu sonucu oluşan iş emirlerini kaydeder."""
        with self._get_connection() as conn:
            df.to_sql("work_orders", conn, if_exists="append", index=False)

    def load_work_orders(self) -> pd.DataFrame:
        """Aktif iş emirlerini sorgular."""
        with self._get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM work_orders ORDER BY day_index ASC", conn)

    def clear_all_tables(self) -> None:
        """Simülasyon sıfırlamaları için tüm tabloları temizler."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for table in ["telemetry", "residuals", "health_rul", "work_orders"]:
                cursor.execute(f"DELETE FROM {table}")
            conn.commit()


if __name__ == "__main__":
    from generator import TelemetryGenerator

    print("Veri tabanı katmanı test ediliyor...")
    db = DatabaseHandler()
    db.clear_all_tables()

    generator = TelemetryGenerator()
    print("1 saatlik örnek telemetri üretiliyor...")
    df_sample = generator.generate_golden_baseline(hours=1.0, sampling_rate_sec=10)

    print("Telemetri veritabanına kaydediliyor...")
    db.save_telemetry(df_sample)

    print("Veritabanından geri okunuyor...")
    df_read = db.load_telemetry(equipment_id="IS_DRIVE_04")
    
    print(f"\n--- IS_DRIVE_04 İçin Okunan Veri Özeti ---")
    print(f"Toplam Satır Sayısı: {len(df_read)}")
    print(f"Sütunlar: {list(df_read.columns)}")
    print(df_read.head(3))
    print("\nVeri tabanı katmanı testi başarıyla tamamlandı!")