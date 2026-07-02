import math
import asyncio
import os
import json
import io
import subprocess
import logging
from enum import IntEnum
from typing import Callable

import numpy as np
from numba import njit
import requests
import rasterio
from PIL import Image

from ..config import settings

logger = logging.getLogger(__name__)

OPENTOPOGRAPHY_API_KEY = settings.opentopography_api_key or os.environ.get("OPENTOPOGRAPHY_API_KEY", "")
if not OPENTOPOGRAPHY_API_KEY:
    logger.warning(
        "OPENTOPOGRAPHY_API_KEY no configurada. "
        "La descarga de DEM fallará. "
        "Configura la variable de entorno OPENTOPOGRAPHY_API_KEY."
    )
WINDNINJA_CLI_PATH = settings.windninja_cli_path

DEM_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DEM_PATH = os.path.join(DEM_DIR, "elevacion.tif")
BOUNDS_PATH = os.path.join(DEM_DIR, "elevacion_bounds.json")
GREENNESS_PATH = os.path.join(DEM_DIR, "vegetacion.json")
GREENNESS_NPY_PATH = os.path.join(DEM_DIR, "vegetacion.npy.npz")

DEFAULT_HUMIDITY = 0.4
DEFAULT_ELEVATION = 1000.0

# Grid cell status as integers for NumPy/Numba compatibility
COMBUSTIBLE = 0
FUEGO = 1
QUEMADO = 2

# FUEL properties as flat arrays for Numba
FUEL_IDS = np.array([1, 4, 8, 10], dtype=np.int32)
FUEL_W0 = np.array([0.162, 1.120, 0.448, 1.100], dtype=np.float64)
FUEL_DELTA = np.array([0.30, 1.80, 0.06, 0.30], dtype=np.float64)
FUEL_SIGMA = np.array([6500.0, 4900.0, 5700.0, 4800.0], dtype=np.float64)
FUEL_MX = np.array([0.20, 0.20, 0.30, 0.25], dtype=np.float64)

# Precomputed wind constants per fuel type
FUEL_WIND_RHO_B = FUEL_W0 / FUEL_DELTA
FUEL_WIND_BETA = FUEL_WIND_RHO_B / 513.0
FUEL_WIND_BETA_OP = 3.348 * (FUEL_SIGMA ** -0.8189)
FUEL_WIND_C = 7.47 * np.exp(-0.133 * (FUEL_SIGMA ** 0.55))
FUEL_WIND_B = 0.02526 * (FUEL_SIGMA ** 0.54)
FUEL_WIND_E = 0.715 * np.exp(-3.59e-4 * FUEL_SIGMA)

FUEL_BURN_MAX = np.array([0, 15, 0, 0, 15, 0, 0, 0, 30, 0, 60], dtype=np.int32)
FUEL_SPREAD_BURN_MAX = np.array([0, 6, 0, 0, 15, 0, 0, 0, 12, 0, 24], dtype=np.int32)


def _fuel_index(fuel_id: int) -> int:
    """Map fuel_id (1,4,8,10) to array index (0,1,2,3)."""
    if fuel_id == 1:
        return 0
    elif fuel_id == 4:
        return 1
    elif fuel_id == 8:
        return 2
    elif fuel_id == 10:
        return 3
    return 0


# ======================================================================
# Numba-optimized physics functions (standalone, no self)
# ======================================================================

@njit(cache=True)
def _jit_fine_fuel_moisture(
    humidity: float, temperature: float, aspect: float
) -> float:
    m_f = 0.03 + 0.261 * humidity - 0.00078 * humidity * temperature
    if 135.0 <= aspect <= 225.0:
        m_f -= 0.02
    elif aspect >= 315.0 or aspect <= 45.0:
        m_f += 0.01
    return max(0.01, min(m_f, 0.50))


@njit(cache=True)
def _jit_rothermel_ros(
    fuel_w0: float, fuel_delta: float, fuel_sigma: float, fuel_Mx: float,
    rho_b: float, beta: float, beta_op: float,
    C_val: float, B_val: float, E_val: float,
    m_f: float, slope: float, u_kph: float,
) -> tuple:
    h = 18600.0
    se = 0.01

    if m_f >= fuel_Mx:
        return (0.0, 0.0, 0.0)

    r_m = m_f / fuel_Mx
    eta_m = 1.0 - 2.59 * r_m + 5.11 * (r_m ** 2) - 3.52 * (r_m ** 3)
    eta_m = max(0.0, min(eta_m, 1.0))

    eta_s = 0.174 * (se ** -0.19)
    eta_s = max(0.0, min(eta_s, 1.0))

    sigma_ratio = fuel_sigma ** 1.5
    gamma_prime_max = sigma_ratio / (495.0 + 0.0594 * sigma_ratio)
    A = 133.0 * (fuel_sigma ** -0.7913)
    beta_ratio = beta / beta_op
    gamma_prime = gamma_prime_max * (beta_ratio ** A) * math.exp(A * (1.0 - beta_ratio))

    wn = fuel_w0 * 0.93
    Ir = gamma_prime * wn * h * eta_m * eta_s

    denom = 192.0 + 0.2595 * fuel_sigma
    flux_num = math.exp((0.792 + 0.681 * (fuel_sigma ** 0.5)) * (beta + 0.1))
    xi = flux_num / denom

    Q_ig = 581.0 + 2594.0 * m_f
    epsilon = math.exp(-138.0 / fuel_sigma)

    phi_S = 5.275 * (beta ** -0.3) * (slope ** 2)

    u_mmin = (u_kph * 1000.0) / 60.0
    u_flame = u_mmin * 0.4
    phi_W = C_val * (u_flame ** B_val) * ((beta / beta_op) ** -E_val)

    R_no_wind_no_slope = (Ir * xi) / (rho_b * epsilon * Q_ig)
    R_head = R_no_wind_no_slope * (1.0 + phi_W + phi_S)
    R_head = max(0.0, R_head)

    Ib = Ir * (R_head / 60.0) * (12.6 / fuel_sigma)
    L = 0.0775 * (Ib ** 0.46) if Ib > 0 else 0.0

    return (R_head, L, Ib)


@njit(cache=True)
def _jit_spread_rothermel(
    r: int, c: int, nr: int, nc: int, neighbor_angle: float,
    humidity: float, temperature: float, detection_time: float, simulation_time: float,
    wind_speed_scalar: float, wind_direction_scalar: float,
    elevation_grid: np.ndarray, slope_grid: np.ndarray, aspect_grid: np.ndarray,
    fuel_grid: np.ndarray, greenness_grid: np.ndarray,
    wind_speed_grid: np.ndarray, wind_direction_grid: np.ndarray,
    extinguished_grid: np.ndarray,
    rows: int, cols: int,
) -> tuple:
    """Numba-JIT spread probability calculation."""
    aspect_val = aspect_grid[r, c] if aspect_grid.size > 0 else 0.0
    m_f = _jit_fine_fuel_moisture(humidity, temperature, aspect_val)

    fuel_id = fuel_grid[r, c]
    fi = 0
    if fuel_id == 4:
        fi = 1
    elif fuel_id == 8:
        fi = 2
    elif fuel_id == 10:
        fi = 3

    R_head, L, Ib = _jit_rothermel_ros(
        FUEL_W0[fi], FUEL_DELTA[fi], FUEL_SIGMA[fi], FUEL_MX[fi],
        FUEL_WIND_RHO_B[fi], FUEL_WIND_BETA[fi], FUEL_WIND_BETA_OP[fi],
        FUEL_WIND_C[fi], FUEL_WIND_B[fi], FUEL_WIND_E[fi],
        m_f, slope_grid[r, c], wind_speed_grid[r, c],
    )

    if R_head <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)

    local_speed = wind_speed_grid[r, c] if wind_speed_grid.size > 0 else wind_speed_scalar
    local_dir = wind_direction_grid[r, c] if wind_direction_grid.size > 0 else wind_direction_scalar
    theta_w = (local_dir + 180.0) % 360.0
    theta_s = aspect_grid[r, c] if aspect_grid.size > 0 else 0.0

    beta = FUEL_WIND_BETA[fi]
    beta_op = FUEL_WIND_BETA_OP[fi]
    C_val = FUEL_WIND_C[fi]
    B_val = FUEL_WIND_B[fi]
    E_val = FUEL_WIND_E[fi]
    slope_val = slope_grid[r, c]

    phi_S = 5.275 * (beta ** -0.3) * (slope_val ** 2)
    u_mmin = (local_speed * 1000.0) / 60.0
    u_flame = u_mmin * 0.4
    phi_W = C_val * (u_flame ** B_val) * ((beta / beta_op) ** -E_val)

    Vx = phi_W * math.sin(math.radians(theta_w)) + phi_S * math.sin(math.radians(theta_s))
    Vy = -phi_W * math.cos(math.radians(theta_w)) - phi_S * math.cos(math.radians(theta_s))

    phi_eff = math.hypot(Vx, Vy)
    if phi_eff > 1e-5:
        theta_eff = math.degrees(math.atan2(Vx, -Vy))
        theta_eff = (theta_eff + 360.0) % 360.0
    else:
        theta_eff = theta_w

    denom_sum = 1.0 + phi_W + phi_S
    R_no = R_head / denom_sum if denom_sum > 0.0 else 0.0
    R_back = R_no

    a = (R_head + R_back) / 2.0
    e = (R_head - R_back) / (R_head + R_back) if (R_head + R_back) > 0.0 else 0.0

    angle_diff = (neighbor_angle - theta_eff + 360.0) % 360.0
    cos_val = math.cos(math.radians(angle_diff))
    denom2 = 1.0 - e * cos_val
    if denom2 > 0.0:
        R_i = (a * (1.0 - e * e)) / denom2
    else:
        R_i = R_back

    is_diag = abs(nr - r) == 1 and abs(nc - c) == 1
    d_i = 21.21 if is_diag else 15.0

    dt = 1.0
    prob = 1.0 - math.exp(-R_i * dt / d_i)

    if extinguished_grid[nr, nc]:
        prob = 0.0

    return (prob, R_head, L, Ib)


# ======================================================================
# SimulationEngine class
# ======================================================================

class SimulationEngine:
    GRID_SIZE = settings.grid_size

    def __init__(self) -> None:
        self.rows: int = 0
        self.cols: int = 0
        self.running = False
        self.paused = False
        self.speed_multiplier: float = 1.0
        self._send_callback: Callable | None = None
        self._task: asyncio.Task | None = None
        self._zone_bounds: tuple[float, float, float, float] = (0, 0, 0, 0)
        self.elevation_grid: np.ndarray = np.empty(0, dtype=np.float64)
        self.slope_grid: np.ndarray = np.empty(0, dtype=np.float64)
        self.aspect_grid: np.ndarray = np.empty(0, dtype=np.float64)
        self.fuel_grid: np.ndarray = np.empty(0, dtype=np.int32)
        self.extinguished_grid: np.ndarray = np.empty(0, dtype=np.bool_)
        self.wind_speed: float = 0
        self.wind_direction: float = 0
        self._humidity: float = DEFAULT_HUMIDITY
        self.temperature: float = 20.0
        self.detection_time: float = 30.0
        self.simulation_time: float = 0.0
        self.wind_speed_grid: np.ndarray = np.empty(0, dtype=np.float64)
        self.wind_direction_grid: np.ndarray = np.empty(0, dtype=np.float64)
        self.greenness_grid: np.ndarray = np.empty(0, dtype=np.float64)
        self.burn_remaining: np.ndarray = np.empty(0, dtype=np.int32)
        self.active_fire_count: int = 0
        self.grid: np.ndarray = np.empty(0, dtype=np.int32)
        self._rng = np.random.default_rng()

    async def configure(
        self,
        wind_speed: float,
        wind_direction: float,
        ignition_lat: float,
        ignition_lng: float,
        zone_coords: list[list[float]],
        send_callback: Callable,
        humidity: float = DEFAULT_HUMIDITY,
        temperature: float = 20.0,
        detection_time: float = 30.0,
        speed_multiplier: float = 1.0,
    ) -> None:
        logger.info("configure() iniciado, zone_coords=%d", len(zone_coords))
        self._compute_zone_bounds(zone_coords)
        logger.info("zone_bounds=%s rows=%d cols=%d", self._zone_bounds, self.rows, self.cols)
        await asyncio.to_thread(self._download_elevation_dem)
        self._load_real_elevation_dem()
        self._compute_slope_and_aspect()
        await self._compute_vegetation_greenness()
        self._compute_fuels()
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self._humidity = humidity
        self.temperature = temperature
        self.detection_time = detection_time
        self.speed_multiplier = speed_multiplier
        self.simulation_time = 0.0
        await self._run_windninja_async()
        self._build_grid()
        self._send_callback = send_callback

        row, col = self._geo_to_grid(ignition_lat, ignition_lng)
        logger.info("Ignición en grid[%d,%d] (lat=%.4f, lng=%.4f)", row, col, ignition_lat, ignition_lng)
        self.grid[row, col] = FUEGO
        self.active_fire_count = 1
        fuel_id = int(self.fuel_grid[row, col])
        max_burn = int(FUEL_BURN_MAX[fuel_id]) if fuel_id < len(FUEL_BURN_MAX) else 15
        self.burn_remaining[row, col] = int(self._rng.integers(10, max_burn + 1))
        logger.info("configure() completado, grid shape=%s", self.grid.shape)

        if self._send_callback:
            self._send_callback([{
                "row": row,
                "col": col,
                "status": "fuego",
                "ros": 0.0,
                "flameLength": 0.0,
                "intensity": 0.0,
                "simulationTime": 0.0,
                "fuelId": int(fuel_id),
            }])

    def set_speed(self, speed: float) -> None:
        self.speed_multiplier = speed

    # ------------------------------------------------------------------
    # Elevation DEM download & loading
    # ------------------------------------------------------------------

    def _download_elevation_dem(self) -> None:
        if os.path.exists(DEM_PATH) and os.path.exists(BOUNDS_PATH):
            try:
                with open(BOUNDS_PATH, "r") as f:
                    saved = json.load(f)
                saved_bounds = (
                    saved["lat_min"],
                    saved["lat_max"],
                    saved["lng_min"],
                    saved["lng_max"],
                )
                if all(
                    abs(a - b) < 1e-5
                    for a, b in zip(saved_bounds, self._zone_bounds)
                ):
                    logger.info("DEM ya descargado y bounds coinciden, omitiendo descarga")
                    return
                else:
                    logger.info("Bounds cambiaron, redescargando DEM...")
            except Exception:
                logger.warning("Error leyendo bounds JSON, redescargando...")
        else:
            logger.info("DEM no encontrado, descargando...")

        os.makedirs(DEM_DIR, exist_ok=True)

        lat_min, lat_max, lng_min, lng_max = self._zone_bounds
        params = {
            "demtype": "SRTMGL1",
            "south": lat_min,
            "north": lat_max,
            "west": lng_min,
            "east": lng_max,
            "outputFormat": "GTiff",
            "apikey": OPENTOPOGRAPHY_API_KEY,
        }

        try:
            resp = requests.get(
                "https://portal.opentopography.org/API/globaldem",
                params=params,
                stream=True,
                timeout=60,
            )
            if resp.status_code != 200:
                logger.error(
                    "Error al descargar DEM: HTTP %s - %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return

            with open(DEM_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            with open(BOUNDS_PATH, "w") as f:
                json.dump(
                    {
                        "lat_min": lat_min,
                        "lat_max": lat_max,
                        "lng_min": lng_min,
                        "lng_max": lng_max,
                    },
                    f,
                )

            logger.info("DEM descargado exitosamente en %s", DEM_PATH)
        except requests.RequestException as e:
            logger.error("Error de conexión al descargar DEM: %s", e)

    def _load_real_elevation_dem(self) -> None:
        self.elevation_grid = np.zeros((self.rows, self.cols), dtype=np.float64)

        if not os.path.exists(DEM_PATH):
            logger.info("Archivo DEM no encontrado, usando elevación plana")
            return

        try:
            with rasterio.open(DEM_PATH) as dataset:
                nodata = dataset.nodata

                lat_min, lat_max, lng_min, lng_max = self._zone_bounds
                lat_step = (lat_max - lat_min) / self.rows
                lng_step = (lng_max - lng_min) / self.cols

                # Vectorized coordinate arrays
                lats = lat_max - (np.arange(self.rows) + 0.5) * lat_step
                lngs = lng_min + (np.arange(self.cols) + 0.5) * lng_step

                # Create meshgrid of all coordinates
                lng_grid, lat_grid = np.meshgrid(lngs, lats)
                coords = np.stack([lng_grid.ravel(), lat_grid.ravel()], axis=1)

                # Sample all points at once using rasterio.sample
                # This is much faster than individual dataset.index calls
                samples = list(dataset.sample(coords, indexes=1))
                values = np.array([s[0] for s in samples], dtype=np.float64).reshape(self.rows, self.cols)

                # Handle nodata
                if nodata is not None:
                    values = np.where(values != nodata, values, 0.0)

                self.elevation_grid = values

            logger.info("DEM cargado en elevation_grid (%dx%d)", self.rows, self.cols)
        except Exception as e:
            logger.error("Error al leer el DEM: %s", e)

    # ------------------------------------------------------------------
    # WindNinja simulation & fallback
    # ------------------------------------------------------------------

    def _locate_output(self, base: str, suffixes: list[str]) -> str | None:
        for suffix in suffixes:
            path = base + suffix
            if os.path.exists(path):
                return path
        return None

    def _read_grid_from_raster(self, path: str, grid: np.ndarray) -> None:
        try:
            with rasterio.open(path) as dataset:
                nodata = dataset.nodata
                lat_min, lat_max, lng_min, lng_max = self._zone_bounds
                lat_step = (lat_max - lat_min) / self.rows
                lng_step = (lng_max - lng_min) / self.cols

                lats = lat_max - (np.arange(self.rows) + 0.5) * lat_step
                lngs = lng_min + (np.arange(self.cols) + 0.5) * lng_step
                lng_grid, lat_grid = np.meshgrid(lngs, lats)
                coords = np.stack([lng_grid.ravel(), lat_grid.ravel()], axis=1)

                samples = list(dataset.sample(coords, indexes=1))
                values = np.array([s[0] for s in samples], dtype=np.float64).reshape(self.rows, self.cols)

                if nodata is not None:
                    values = np.where(values != nodata, values, 0.0)

                grid[:, :] = values
        except Exception as e:
            logger.warning("Error leyendo raster %s: %s", path, e)

    async def _run_windninja_async(self) -> None:
        self.wind_speed_grid = np.full(
            (self.rows, self.cols), self.wind_speed, dtype=np.float64
        )
        self.wind_direction_grid = np.full(
            (self.rows, self.cols), self.wind_direction, dtype=np.float64
        )

        if not os.path.exists(DEM_PATH):
            logger.info("DEM no disponible, omitiendo WindNinja")
            return

        def _run_windninja_sync():
            try:
                cmd = [
                    WINDNINJA_CLI_PATH,
                    "--elevation_file", DEM_PATH,
                    "--input_speed", str(self.wind_speed),
                    "--input_direction", str(self.wind_direction),
                    "--output_speed_units", "kph",
                    "--mesh_resolution", "100",
                    "--vegetation", "grass",
                    "--num_threads", "4",
                ]
                logger.info("Ejecutando WindNinja: %s", " ".join(cmd))
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    logger.warning("WindNinja retornó código %d: %s", result.returncode, result.stderr[:300])
                    raise RuntimeError("WindNinja falló")

                logger.info("WindNinja completado exitosamente")

                base = os.path.splitext(DEM_PATH)[0]
                vel_path = self._locate_output(base, [
                    "_vel.tif", "_speed.tif", "_vel.asc", "_speed.asc",
                ])
                ang_path = self._locate_output(base, [
                    "_ang.tif", "_direction.tif", "_ang.asc", "_direction.asc",
                ])

                if vel_path and ang_path:
                    wind_speed_grid = np.zeros((self.rows, self.cols), dtype=np.float64)
                    wind_direction_grid = np.zeros((self.rows, self.cols), dtype=np.float64)
                    self._read_grid_from_raster(vel_path, wind_speed_grid)
                    self._read_grid_from_raster(ang_path, wind_direction_grid)
                    logger.info("Viento local cargado desde WindNinja")
                    return wind_speed_grid, wind_direction_grid
                else:
                    logger.warning("No se encontraron archivos de salida de WindNinja, usando fallback")
                    return None, None

            except FileNotFoundError:
                logger.warning("WindNinja no está instalado en %s", WINDNINJA_CLI_PATH)
                return None, None
            except (subprocess.TimeoutExpired, OSError, Exception) as e:
                logger.warning("WindNinja no disponible (%s), usando fallback matemático", e)
                return None, None

        wind_speed_grid, wind_direction_grid = await asyncio.to_thread(_run_windninja_sync)

        if wind_speed_grid is not None and wind_direction_grid is not None:
            self.wind_speed_grid = wind_speed_grid
            self.wind_direction_grid = wind_direction_grid
        else:
            self._apply_wind_fallback()

    def _apply_wind_fallback(self) -> None:
        lat_min, lat_max, lng_min, lng_max = self._zone_bounds
        lat_step = (lat_max - lat_min) / self.rows
        lng_step = (lng_max - lng_min) / self.cols
        lats = lat_max - (np.arange(self.rows) + 0.5) * lat_step
        lngs = lng_min + (np.arange(self.cols) + 0.5) * lng_step

        for r in range(self.rows):
            elev_row = self.elevation_grid[r, :]
            factor = 1.0 + (elev_row - 1000.0) / 1000.0
            self.wind_speed_grid[r, :] = np.maximum(0.0, self.wind_speed * factor)
            self.wind_direction_grid[r, :] = self.wind_direction

    # ------------------------------------------------------------------
    # Zone bounds & grid
    # ------------------------------------------------------------------

    def _compute_zone_bounds(self, coords: list[list[float]]) -> None:
        lats = [p[0] for p in coords]
        lngs = [p[1] for p in coords]
        self._zone_bounds = (
            min(lats),
            max(lats),
            min(lngs),
            max(lngs),
        )
        lat_span = self._zone_bounds[1] - self._zone_bounds[0]
        lng_span = self._zone_bounds[3] - self._zone_bounds[2]
        if lat_span == 0:
            self._zone_bounds = (
                self._zone_bounds[0] - 0.001,
                self._zone_bounds[1] + 0.001,
                self._zone_bounds[2],
                self._zone_bounds[3],
            )
            lat_span = 0.002
        if lng_span == 0:
            self._zone_bounds = (
                self._zone_bounds[0],
                self._zone_bounds[1],
                self._zone_bounds[2] - 0.001,
                self._zone_bounds[3] + 0.001,
            )
            lng_span = 0.002

        lat_avg = (self._zone_bounds[0] + self._zone_bounds[1]) / 2.0
        lat_span_m = lat_span * 111139.0
        lng_span_m = lng_span * 111139.0 * math.cos(math.radians(lat_avg))

        if lat_span_m >= lng_span_m:
            self.rows = self.GRID_SIZE
            self.cols = max(1, int(round(self.GRID_SIZE * (lng_span_m / lat_span_m))))
        else:
            self.cols = self.GRID_SIZE
            self.rows = max(1, int(round(self.GRID_SIZE * (lat_span_m / lng_span_m))))

    # ------------------------------------------------------------------
    # Satellite greenness (vegetation from Esri World Imagery)
    # ------------------------------------------------------------------

    def _lat_lng_to_tile(self, lat: float, lng: float, zoom: int) -> tuple[int, int]:
        n = 2.0 ** zoom
        x_tile = int((lng + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y_tile = int(
            (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
            / 2.0
            * n
        )
        return x_tile, y_tile

    def _geo_to_tile_pixel(
        self, lat: float, lng: float, zoom: int
    ) -> tuple[int, int, int, int]:
        n = 2.0 ** zoom
        x_pixel = (lng + 180.0) / 360.0 * n * 256.0
        lat_rad = math.radians(lat)
        y_pixel = (
            (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
            / 2.0
            * n
            * 256.0
        )
        tx = int(x_pixel) // 256
        ty = int(y_pixel) // 256
        px = int(x_pixel) % 256
        py = int(y_pixel) % 256
        return tx, ty, px, py

    def _pick_zoom_level(self) -> int:
        lat_min, lat_max, lng_min, lng_max = self._zone_bounds
        lng_span = lng_max - lng_min
        if lng_span < 1e-8:
            lng_span = 0.001
        target_px_per_cell = 6.0
        z = math.log2(target_px_per_cell * 360.0 * self.cols / (lng_span * 256.0))
        return max(12, min(18, int(round(z))))

    async def _compute_vegetation_greenness(self) -> None:
        self.greenness_grid = np.zeros((self.rows, self.cols), dtype=np.float64)

        # Try loading from compressed .npz first (fastest)
        if os.path.exists(GREENNESS_NPY_PATH):
            try:
                cached = np.load(GREENNESS_NPY_PATH, allow_pickle=True)
                cached_bounds = tuple(cached["bounds"].tolist())
                if all(abs(a - b) < 1e-5 for a, b in zip(cached_bounds, self._zone_bounds)):
                    cached_grid = cached["grid"]
                    if cached_grid.shape == (self.rows, self.cols):
                        self.greenness_grid = cached_grid.astype(np.float64)
                        logger.info(
                            "Greenness cargado desde .npz (%d celdas)",
                            self.greenness_grid.size,
                        )
                        return
                    else:
                        logger.warning("Dimensiones de cache .npz no coinciden (%s vs %dx%d), recalculando...",
                                        cached_grid.shape, self.rows, self.cols)
            except Exception:
                logger.warning("Error leyendo cache .npz, intentando JSON...")

        # Fallback to JSON
        if os.path.exists(GREENNESS_PATH):
            try:
                with open(GREENNESS_PATH, "r") as f:
                    cached = json.load(f)
                if cached.get("bounds") == list(self._zone_bounds):
                    cached_grid = cached.get("grid", [])
                    if len(cached_grid) == self.rows and cached_grid and len(cached_grid[0]) == self.cols:
                        self.greenness_grid = np.array(cached_grid, dtype=np.float64)
                        logger.info(
                            "Greenness cargado desde cache JSON (%d celdas)",
                            self.greenness_grid.size,
                        )
                        return
                    else:
                        logger.warning("Dimensiones de cache de greenness no coinciden, recalculando...")
            except Exception:
                logger.warning("Error leyendo cache de greenness, recalculando...")

        zoom = self._pick_zoom_level()
        needed: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}

        lat_min, lat_max, lng_min, lng_max = self._zone_bounds
        lat_step = (lat_max - lat_min) / self.rows
        lng_step = (lng_max - lng_min) / self.cols

        for r in range(self.rows):
            lat = lat_max - (r + 0.5) * lat_step
            for c in range(self.cols):
                lng = lng_min + (c + 0.5) * lng_step
                tx, ty, px, py = self._geo_to_tile_pixel(lat, lng, zoom)
                key = (tx, ty)
                if key not in needed:
                    needed[key] = []
                needed[key].append((r, c, px, py))

        if len(needed) > 8:
            logger.warning(
                "Demasiados tiles satelitales requeridos (%d). "
                "Para evitar bloqueos de API y lentitud, se usa fallback basado en elevación.",
                len(needed)
            )
            self._greenness_fallback()
            self._cache_greenness()
            return

        logger.info(
            "Descargando %d tile(s) satelitales (zoom %d) para greenness...",
            len(needed),
            zoom,
        )

        import aiohttp
        sem = asyncio.Semaphore(8)
        tiles: dict[tuple[int, int], Image.Image] = {}

        async def fetch_tile(session: aiohttp.ClientSession, tx: int, ty: int):
            url = (
                f"https://server.arcgisonline.com/ArcGIS/rest/services"
                f"/World_Imagery/MapServer/tile/{zoom}/{ty}/{tx}"
            )
            async with sem:
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            img = Image.open(io.BytesIO(content)).convert("RGB")
                            return (tx, ty), img
                except Exception as e:
                    logger.warning("Error descargando tile %d,%d: %s", tx, ty, e)
                return (tx, ty), None

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_tile(session, tx, ty) for tx, ty in needed.keys()]
            results = await asyncio.gather(*tasks)

        for (tx, ty), img in results:
            if img is not None:
                tiles[(tx, ty)] = img

        if not tiles:
            logger.warning(
                "No se pudieron descargar tiles satelitales, "
                "usando greenness basado en elevación"
            )
            self._greenness_fallback()
            self._cache_greenness()
            return

        for (tx, ty), cells in needed.items():
            img = tiles.get((tx, ty))
            if img is None:
                continue
            for r, c, px, py in cells:
                if 0 <= px < 256 and 0 <= py < 256:
                    R, G, B = img.getpixel((px, py))
                    green = max(0.0, (G - R) / (R + G + B + 1) * 3.0)
                    self.greenness_grid[r, c] = min(1.0, green)

        self._cache_greenness()
        sampled = int(np.count_nonzero(self.greenness_grid))
        logger.info(
            "Greenness calculado desde satélite (%d celdas con vegetación, zoom %d)",
            sampled,
            zoom,
        )

    def _greenness_fallback(self) -> None:
        elev = self.elevation_grid
        self.greenness_grid = np.where(
            elev >= 2000, 0.15,
            np.where(elev >= 1500, 0.30,
            np.where(elev >= 1000, 0.50,
            np.where(elev >= 500, 0.65, 0.80)))
        )

    def _cache_greenness(self) -> None:
        try:
            os.makedirs(DEM_DIR, exist_ok=True)
            # Save as compressed .npz (fast load, small size)
            np.savez_compressed(
                GREENNESS_NPY_PATH,
                bounds=np.array(self._zone_bounds, dtype=np.float64),
                grid=self.greenness_grid.astype(np.float32),
            )
            logger.info("Greenness cacheado en %s", GREENNESS_NPY_PATH)
        except Exception as e:
            logger.warning("Error cacheando greenness: %s", e)

    def _compute_slope_and_aspect(self) -> None:
        lat_min, lat_max, lng_min, lng_max = self._zone_bounds
        lat_avg = (lat_min + lat_max) / 2.0
        lat_dist = 111139.0
        lng_dist = 111139.0 * math.cos(math.radians(lat_avg))

        lat_span_m = (lat_max - lat_min) * lat_dist
        lng_span_m = (lng_max - lng_min) * lng_dist

        dx = max(1.0, lng_span_m / self.cols)
        dy = max(1.0, lat_span_m / self.rows)

        elev = self.elevation_grid

        # Compute gradients using numpy (central differences)
        dz_dy, dz_dx = np.gradient(elev, dy, dx)

        self.slope_grid = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
        angle_deg = np.degrees(np.arctan2(dz_dy, dz_dx))
        self.aspect_grid = (90.0 - angle_deg + 360.0) % 360.0

    def _compute_fuels(self) -> None:
        g = self.greenness_grid
        self.fuel_grid = np.where(
            g > 0.6, 10,
            np.where(g > 0.4, 8,
            np.where(g > 0.15, 4, 1))
        ).astype(np.int32)

    def _build_grid(self) -> None:
        # Double-buffered grids: self.grid (current) and self._next_grid (next step)
        self.grid = np.full((self.rows, self.cols), COMBUSTIBLE, dtype=np.int32)
        self._next_grid = np.full((self.rows, self.cols), COMBUSTIBLE, dtype=np.int32)
        self.burn_remaining = np.zeros((self.rows, self.cols), dtype=np.int32)
        self._next_burn = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.extinguished_grid = np.zeros((self.rows, self.cols), dtype=np.bool_)
        self.active_fire_count = 0

    def _geo_to_grid(self, lat: float, lng: float) -> tuple[int, int]:
        lat_min, lat_max, lng_min, lng_max = self._zone_bounds
        lat_span = lat_max - lat_min
        lng_span = lng_max - lng_min
        row = int(((lat_max - lat) / lat_span) * self.rows)
        col = int(((lng - lng_min) / lng_span) * self.cols)
        row = max(0, min(row, self.rows - 1))
        col = max(0, min(col, self.cols - 1))
        return row, col

    def _grid_to_geo(self, row: int, col: int) -> tuple[float, float]:
        lat_min, lat_max, lng_min, lng_max = self._zone_bounds
        lat_step = (lat_max - lat_min) / self.rows
        lng_step = (lng_max - lng_min) / self.cols
        lat = lat_max - (row + 0.5) * lat_step
        lng = lng_min + (col + 0.5) * lng_step
        return lat, lng

    # ------------------------------------------------------------------
    # INP physical model helpers
    # ------------------------------------------------------------------

    def _get_elevation(self, r: int, c: int) -> float:
        if self.elevation_grid.size > 0:
            try:
                return float(self.elevation_grid[r, c])
            except IndexError:
                pass
        return DEFAULT_ELEVATION

    def _get_slope_factor(self, r1: int, c1: int, r2: int, c2: int) -> float:
        e1 = self._get_elevation(r1, c1)
        e2 = self._get_elevation(r2, c2)
        elev_diff = e2 - e1
        lat_min, lat_max, lng_min, lng_max = self._zone_bounds
        lat_per_cell = (lat_max - lat_min) / self.rows
        lng_per_cell = (lng_max - lng_min) / self.cols
        dr = (r2 - r1) * lat_per_cell
        dc = (c2 - c1) * lng_per_cell
        distance = math.hypot(dr, dc)
        if distance < 1e-10:
            return 1.0
        slope_ratio = elev_diff / distance
        P = 1.0 + slope_ratio * 3.0
        return max(0.3, min(P, 3.0))

    def _get_vegetation_params(self, r: int, c: int) -> tuple[float, float, float]:
        g = 0.5
        if self.greenness_grid.size > 0:
            try:
                g = float(self.greenness_grid[r, c])
            except IndexError:
                pass
        if g > 0.60:
            return (1.2, 0.3, 0.05)
        elif g > 0.40:
            return (1.1, 0.5, 0.15)
        elif g > 0.25:
            return (1.0, 0.7, 0.30)
        elif g > 0.10:
            return (0.8, 0.9, 0.50)
        else:
            return (0.5, 1.1, 0.80)

    def _calculate_rothermel_ros(self, r: int, c: int, m_f: float) -> tuple[float, float, float]:
        fuel_id = int(self.fuel_grid[r, c]) if self.fuel_grid.size > 0 else 1
        fi = _fuel_index(fuel_id)

        R_head, L, Ib = _jit_rothermel_ros(
            FUEL_W0[fi], FUEL_DELTA[fi], FUEL_SIGMA[fi], FUEL_MX[fi],
            FUEL_WIND_RHO_B[fi], FUEL_WIND_BETA[fi], FUEL_WIND_BETA_OP[fi],
            FUEL_WIND_C[fi], FUEL_WIND_B[fi], FUEL_WIND_E[fi],
            m_f,
            float(self.slope_grid[r, c]) if self.slope_grid.size > 0 else 0.0,
            float(self.wind_speed_grid[r, c]) if self.wind_speed_grid.size > 0 else self.wind_speed,
        )
        return (float(R_head), float(L), float(Ib))

    def _spread_probability_rothermel(
        self,
        r: int,
        c: int,
        nr: int,
        nc: int,
        neighbor_angle: float,
    ) -> tuple[float, float, float, float]:
        return _jit_spread_rothermel(
            r, c, nr, nc, neighbor_angle,
            self._humidity, self.temperature,
            self.detection_time, self.simulation_time,
            self.wind_speed, self.wind_direction,
            self.elevation_grid, self.slope_grid, self.aspect_grid,
            self.fuel_grid, self.greenness_grid,
            self.wind_speed_grid, self.wind_direction_grid,
            self.extinguished_grid,
            self.rows, self.cols,
        )

    def _spread_probability_inp(
        self,
        r: int,
        c: int,
        nr: int,
        nc: int,
        neighbor_angle: float,
    ) -> float:
        local_speed = float(self.wind_speed_grid[nr, nc]) if self.wind_speed_grid.size > 0 else self.wind_speed
        local_dir = float(self.wind_direction_grid[nr, nc]) if self.wind_direction_grid.size > 0 else self.wind_direction
        K, C, h_extra = self._get_vegetation_params(nr, nc)
        P = self._get_slope_factor(r, c, nr, nc)
        H = max(0.01, self._humidity * 0.35 + h_extra)

        wind_toward = (360 - local_dir) % 360
        angle_diff = (neighbor_angle - wind_toward + 360) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        alignment = math.cos(math.radians(angle_diff))

        if alignment > 0:
            background = 0.12
        else:
            background = 0.04 + 0.04 * (1.0 + alignment)

        if alignment > 0 and local_speed > 0:
            V_eff = local_speed * alignment
            INP = (K * C * P * (V_eff ** 2)) / H
            wind_prob = INP / (INP + 1.0)
        else:
            wind_prob = 0.0
            if alignment < 0 and local_speed > 0:
                V_up = local_speed * (-alignment)
                INP_up = 0.03 * (K * C * P * (V_up ** 2)) / H
                wind_prob = min(0.25, INP_up / (INP_up + 1.0))

        prob = min(background + wind_prob, 0.95)
        g = 0.5
        if self.greenness_grid.size > 0:
            try:
                g = float(self.greenness_grid[nr, nc])
            except IndexError:
                pass
        prob *= max(0.05, g)
        return prob

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.running = True
        self.paused = False

    def pause(self) -> None:
        self.paused = True

    def stop(self) -> None:
        self.running = False
        self.paused = False

    async def run_loop(self) -> None:
        step = 0
        pending_updates: list[dict] = []
        logger.info("run_loop iniciado, active_fire_count=%d", self.active_fire_count)
        try:
            while self.running:
                if self.paused:
                    await asyncio.sleep(0.5)
                    continue

                delay = max(0.01, 0.8 / self.speed_multiplier)
                await asyncio.sleep(delay)
                self.simulation_time += 1.0

                updates = self._step()
                if self._send_callback:
                    pending_updates.extend(updates)

                    send_threshold = 2 if self.active_fire_count > 500 else 1
                    if step % send_threshold == 0 and pending_updates:
                        self._send_callback(pending_updates)
                        pending_updates = []

                if step < 5 or step % 50 == 0:
                    logger.info(
                        "step=%d time=%.0f fires=%d updates=%d",
                        step, self.simulation_time, self.active_fire_count, len(updates),
                    )

                if not self._has_burning():
                    if pending_updates and self._send_callback:
                        self._send_callback(pending_updates)
                        pending_updates = []
                    logger.info("Sin fuego activo, deteniendo en step=%d", step)
                    self.running = False
                    break

                step += 1
        except Exception as e:
            logger.error("Excepción en run_loop en step=%d: %s", step, e, exc_info=True)

    # ------------------------------------------------------------------
    # Simulation step & propagation
    # ------------------------------------------------------------------

    def _step(self) -> list[dict]:
        self._next_grid[:, :] = self.grid
        self._next_burn[:, :] = self.burn_remaining

        updates: list[dict] = []

        fire_mask = (self.grid == FUEGO)
        fire_positions = np.argwhere(fire_mask)
        n_fires = len(fire_positions)
        if n_fires == 0:
            self.grid, self._next_grid = self._next_grid, self.grid
            self.burn_remaining, self._next_burn = self._next_burn, self.burn_remaining
            return updates

        rng = self._rng
        spread_probs = rng.random(n_fires)
        quick_burn_low = rng.random(n_fires)
        quick_burn_high = rng.random(n_fires)
        for idx in range(n_fires):
            r = int(fire_positions[idx, 0])
            c = int(fire_positions[idx, 1])

            if spread_probs[idx] < 0.55:
                self._spread_to_neighbors(r, c, self._next_grid, self._next_burn, updates)

            g = float(self.greenness_grid[r, c]) if self.greenness_grid.size > 0 else 0.5

            if g < 0.10:
                if quick_burn_low[idx] < 0.95:
                    self._next_grid[r, c] = QUEMADO
                    updates.append({"row": r, "col": c, "status": "quemado"})
                    self._next_burn[r, c] = 0
                    self.active_fire_count -= 1
                    continue
            elif g < 0.25:
                if quick_burn_high[idx] < 0.70:
                    self._next_grid[r, c] = QUEMADO
                    updates.append({"row": r, "col": c, "status": "quemado"})
                    self._next_burn[r, c] = 0
                    self.active_fire_count -= 1
                    continue

            self._next_burn[r, c] -= 1
            if self._next_burn[r, c] <= 0:
                self._next_grid[r, c] = QUEMADO
                updates.append({"row": r, "col": c, "status": "quemado"})
                self.active_fire_count -= 1

        self.grid, self._next_grid = self._next_grid, self.grid
        self.burn_remaining, self._next_burn = self._next_burn, self.burn_remaining

        return updates

    def _spread_to_neighbors(
        self,
        r: int,
        c: int,
        new_grid: np.ndarray,
        new_burn: np.ndarray | None,
        updates: list[dict],
    ) -> None:
        rng = self._rng
        base_dirs = [
            (-1, -1, 225),
            (-1, 0, 180),
            (-1, 1, 135),
            (0, -1, 270),
            (0, 1, 90),
            (1, -1, 315),
            (1, 0, 0),
            (1, 1, 45),
        ]

        rng.shuffle(base_dirs)
        check_count = int(rng.integers(4, 7))

        for dr, dc, angle in base_dirs[:check_count]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                continue
            if new_grid[nr, nc] != COMBUSTIBLE:
                continue

            prob, R_head, L, Ib = self._spread_probability_rothermel(r, c, nr, nc, angle)
            noise = rng.uniform(0.75, 1.25)
            prob = min(prob * noise, 0.95)

            if self.simulation_time > self.detection_time:
                if Ib < 500:
                    prob *= 0.15
                elif Ib < 2000:
                    prob *= 0.50
                else:
                    prob *= 0.95

            if prob > 0 and rng.random() < prob:
                new_grid[nr, nc] = FUEGO
                self.active_fire_count += 1
                fuel_id = int(self.fuel_grid[nr, nc]) if self.fuel_grid.size > 0 else 1
                max_burn = int(FUEL_SPREAD_BURN_MAX[fuel_id]) if fuel_id < len(FUEL_SPREAD_BURN_MAX) else 15
                if new_burn is not None:
                    new_burn[nr, nc] = int(rng.integers(4, max_burn + 1))

                updates.append({
                    "row": nr,
                    "col": nc,
                    "status": "fuego",
                    "ros": float(R_head),
                    "flameLength": float(L),
                    "intensity": float(Ib),
                    "simulationTime": float(self.simulation_time),
                    "fuelId": int(fuel_id)
                })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_burning(self) -> bool:
        return self.active_fire_count > 0

    @property
    def active(self) -> bool:
        return self.running and not self.paused
