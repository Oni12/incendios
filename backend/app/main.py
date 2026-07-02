import json
import asyncio
import logging
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .services.kml_parser import parse_polygons
from .services.simulation_engine import SimulationEngine
from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simulador de Incendios Forestales")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:4201"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEM_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "elevacion.tif")
WINDNINJA_PATH = settings.windninja_cli_path


@app.get("/api/simulation/health")
async def health():
    return {
        "status": "ok",
        "dem_available": os.path.exists(DEM_PATH),
        "api_key_configured": bool(os.environ.get("OPENTOPOGRAPHY_API_KEY", "")),
        "windninja_available": os.path.exists(WINDNINJA_PATH),
    }


@app.get("/api/simulation/initial")
async def get_initial_data():
    try:
        polygons = parse_polygons()
        # Calcular las dimensiones de la matriz dinámica para que sea cuadrada
        temp_engine = SimulationEngine()
        coords = [pt for poly in polygons for pt in poly["coordinates"]]
        temp_engine._compute_zone_bounds(coords)
        rows = temp_engine.rows
        cols = temp_engine.cols
    except Exception as e:
        logger.error("Error parsing KML: %s", e)
        return {"zonePolygons": [], "wind": {"speed": 0, "direction": 0}, "rows": 100, "cols": 100}

    return {
        "zonePolygons": polygons,
        "wind": {"speed": 15.5, "direction": 180},
        "rows": rows,
        "cols": cols,
    }


@app.websocket("/api/simulation/ws")
async def simulation_websocket(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket conectado")

    engine = SimulationEngine()
    send_queue: asyncio.Queue = asyncio.Queue()

    async def send_from_queue():
        while True:
            payload = await send_queue.get()
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                break

    queue_worker = asyncio.create_task(send_from_queue())

    def send_callback(updates: list[dict]) -> None:
        send_queue.put_nowait(updates)

    engine_task: asyncio.Task | None = None

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")

            if action == "start":
                config = msg.get("config", {})
                wind_speed = config.get("windSpeed", 15.5)
                wind_direction = config.get("windDirection", 180)
                humidity = config.get("humidity", 0.4)
                temperature = config.get("temperature", 20.0)
                detection_time = config.get("detectionTime", 30.0)
                speed_multiplier = config.get("simulationSpeed", 1.0)
                ignition = config.get("ignitionPoint", [0, 0])
                ignition_lat, ignition_lng = ignition

                all_polygons = parse_polygons()
                coords = [pt for poly in all_polygons for pt in poly["coordinates"]]

                # Validate ignition point is within zone bounds
                temp_engine = SimulationEngine()
                temp_engine._compute_zone_bounds(coords)
                lat_min, lat_max, lng_min, lng_max = temp_engine._zone_bounds

                if not (lat_min <= ignition_lat <= lat_max):
                    await ws.send_text(json.dumps({
                        "action": "error",
                        "message": f"Punto de ignición fuera de rango latitud ({ignition_lat:.4f}). Rango: [{lat_min:.4f}, {lat_max:.4f}]"
                    }))
                    continue
                if not (lng_min <= ignition_lng <= lng_max):
                    await ws.send_text(json.dumps({
                        "action": "error",
                        "message": f"Punto de ignición fuera de rango longitud ({ignition_lng:.4f}). Rango: [{lng_min:.4f}, {lng_max:.4f}]"
                    }))
                    continue

                try:
                    await engine.configure(
                        wind_speed=wind_speed,
                        wind_direction=wind_direction,
                        ignition_lat=ignition_lat,
                        ignition_lng=ignition_lng,
                        zone_coords=coords,
                        send_callback=send_callback,
                        humidity=humidity,
                        temperature=temperature,
                        detection_time=detection_time,
                        speed_multiplier=speed_multiplier,
                    )
                except Exception as e:
                    logger.error("Error en configure(): %s", e, exc_info=True)
                    await ws.send_text(json.dumps({
                        "action": "error",
                        "message": f"Error de configuración: {e}"
                    }))
                    continue

                engine.start()

                if engine_task and not engine_task.done():
                    engine_task.cancel()
                engine_task = asyncio.create_task(engine.run_loop())

                logger.info(
                    "Simulación iniciada: viento=%s km/h dir=%s° ignición=%s,%s velocidad=%sx rows=%d cols=%d",
                    wind_speed,
                    wind_direction,
                    ignition_lat,
                    ignition_lng,
                    speed_multiplier,
                    engine.rows,
                    engine.cols,
                )

            elif action == "pause":
                engine.pause()
                await ws.send_text(json.dumps({"action": "paused"}))
                logger.info("Simulación pausada")

            elif action == "resume":
                engine.start()
                await ws.send_text(json.dumps({"action": "resumed"}))
                logger.info("Simulación reanudada")

            elif action == "set_speed":
                speed = msg.get("speed", 1.0)
                engine.set_speed(speed)
                logger.info("Velocidad de simulación ajustada a: %sx", speed)

            elif action == "stop":
                engine.stop()
                if engine_task and not engine_task.done():
                    engine_task.cancel()
                    engine_task = None
                await ws.send_text(json.dumps({"action": "stopped"}))
                logger.info("Simulación detenida")

    except WebSocketDisconnect:
        logger.info("WebSocket desconectado")
    except Exception as e:
        logger.error("Error en WebSocket: %s", e)
    finally:
        engine.stop()
        if engine_task and not engine_task.done():
            engine_task.cancel()
        queue_worker.cancel()
        try:
            await ws.close()
        except Exception:
            pass
