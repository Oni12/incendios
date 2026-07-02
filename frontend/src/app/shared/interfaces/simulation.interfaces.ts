export interface WindData {
  speed: number;
  direction: number;
}

export interface ZonePolygon {
  coordinates: [number, number][];
  year?: number | null;
}

export interface SimulationConfig {
  windSpeed: number;
  windDirection: number;
  ignitionPoint: [number, number];
  humidity?: number;
  temperature?: number;
  detectionTime?: number;
  simulationSpeed?: number;
}

export interface CellUpdate {
  row: number;
  col: number;
  status: 'combustible' | 'fuego' | 'quemado';
  ros?: number;
  flameLength?: number;
  intensity?: number;
  simulationTime?: number;
  fuelId?: number;
}

export interface InitialData {
  zonePolygons: ZonePolygon[];
  wind: WindData;
  rows: number;
  cols: number;
}

export type WsClientMessage =
  | { action: 'start'; config: SimulationConfig }
  | { action: 'pause' }
  | { action: 'resume' }
  | { action: 'stop' }
  | { action: 'set_speed'; speed: number };

export type WsServerMessage =
  | CellUpdate[]
  | { action: 'started' }
  | { action: 'paused' }
  | { action: 'resumed' }
  | { action: 'stopped' }
  | { action: 'error'; message: string };

export interface HealthResponse {
  status: string;
  dem_available: boolean;
  api_key_configured: boolean;
  windninja_available: boolean;
}
