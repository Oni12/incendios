import { Component, OnInit, OnDestroy, signal, computed } from '@angular/core';
import { SimulationService } from './services/simulation';
import { ControlPanel } from './components/control-panel/control-panel';
import { MapViewer } from './components/map-viewer/map-viewer';
import { StatsPanel, FuelDist } from './components/stats-panel/stats-panel';
import { CellUpdate, ZonePolygon, WsServerMessage } from '../../shared/interfaces';
import { Subscription } from 'rxjs';
import { filter } from 'rxjs/operators';

@Component({
  selector: 'app-simulator',
  standalone: true,
  imports: [ControlPanel, MapViewer, StatsPanel],
  templateUrl: './simulator.html',
  styleUrls: ['./simulator.css'],
})
export class SimulatorComponent implements OnInit, OnDestroy {
  zonePolygons = signal<ZonePolygon[]>([]);
  cellUpdates = signal<CellUpdate[]>([]);
  ignitionPoint = signal<[number, number] | null>(null);
  windSpeed = signal(15.5);
  windDirection = signal(180);
  humidity = signal(40); // %
  temperature = signal(20); // °C
  detectionTime = signal(30); // min
  simulationSpeed = signal(1); // Multiplicador de velocidad (1x, 2x, etc.)
  rows = signal(100);
  cols = signal(100);
  isRunning = signal(false);
  isPaused = signal(false);
  loading = signal(true);

  // Estadísticas calculadas
  maxRos = signal(0.0);
  burnedArea = signal(0.0);
  suppressionActive = signal(false);
  elapsedTime = signal(0.0);

  // Nuevas estadísticas avanzadas
  maxIntensity = signal(0.0);
  maxFlameLength = signal(0.0);
  activeFiresCount = signal(0);
  burnedCellsCount = signal(0);
  fuelDistribution = signal<FuelDist>({ pastizal: 0, matorral: 0, bosqueRalo: 0, bosqueDenso: 0 });
  showPerimeter = signal(false);
  errorMessage = signal<string | null>(null);

  canStart = computed(() => this.ignitionPoint() !== null && !this.isRunning());

  private subs: Subscription[] = [];
  private burnedOrBurning = new Set<string>();
  private cellStates = new Map<string, { status: string, fuelId: number }>();
  private statsThrottleTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingStatsUpdate = false;

  constructor(private simulation: SimulationService) {}

  ngOnInit(): void {
    this.subs.push(
      this.simulation.getInitialData().subscribe({
        next: (data) => {
          this.zonePolygons.set(data.zonePolygons);
          this.windSpeed.set(data.wind.speed);
          this.windDirection.set(data.wind.direction);
          this.rows.set(data.rows);
          this.cols.set(data.cols);
          this.loading.set(false);
        },
        error: () => this.loading.set(false),
      })
    );

    this.subs.push(
      this.simulation.cellUpdates.subscribe((updates) => {
        this.cellUpdates.set(updates);

        let currentMaxRos = this.maxRos();
        let currentMaxIntensity = this.maxIntensity();
        let currentMaxFlame = this.maxFlameLength();
        let time = this.elapsedTime();

        for (const update of updates) {
          const key = `${update.row},${update.col}`;
          if (update.status === 'fuego') {
            this.cellStates.set(key, { status: 'fuego', fuelId: update.fuelId || 1 });
            this.burnedOrBurning.add(key);
          } else if (update.status === 'quemado') {
            this.cellStates.set(key, { status: 'quemado', fuelId: update.fuelId || 1 });
            this.burnedOrBurning.add(key);
          }
          if (update.ros && update.ros > currentMaxRos) {
            currentMaxRos = update.ros;
          }
          if (update.intensity && update.intensity > currentMaxIntensity) {
            currentMaxIntensity = update.intensity;
          }
          if (update.flameLength && update.flameLength > currentMaxFlame) {
            currentMaxFlame = update.flameLength;
          }
          if (update.simulationTime && update.simulationTime > time) {
            time = update.simulationTime;
          }
        }

        this.maxRos.set(currentMaxRos);
        this.maxIntensity.set(currentMaxIntensity);
        this.maxFlameLength.set(currentMaxFlame);
        this.elapsedTime.set(time);
        this.burnedArea.set(this.burnedOrBurning.size * 0.0225);
        if (time > this.detectionTime()) {
          this.suppressionActive.set(true);
        }

        if (!this.pendingStatsUpdate) {
          this.pendingStatsUpdate = true;
          this.statsThrottleTimer = setTimeout(() => {
            this.pendingStatsUpdate = false;
            this.computeHeavyStats();
          }, 500);
        }
      })
    );

    this.subs.push(
      this.simulation.rawMessages
        .pipe(filter((msg): msg is Extract<WsServerMessage, { action: 'error' }> =>
          !Array.isArray(msg) && 'action' in msg && msg.action === 'error'
        ))
        .subscribe((msg) => {
          this.errorMessage.set(msg.message);
          setTimeout(() => this.errorMessage.set(null), 8000);
        })
    );
  }

  ngOnDestroy(): void {
    this.subs.forEach((s) => s.unsubscribe());
    this.simulation.disconnect();
    if (this.statsThrottleTimer) {
      clearTimeout(this.statsThrottleTimer);
    }
  }

  private computeHeavyStats(): void {
    let activeCount = 0;
    let burnedCount = 0;
    const fuelDist: FuelDist = { pastizal: 0, matorral: 0, bosqueRalo: 0, bosqueDenso: 0 };

    for (const val of this.cellStates.values()) {
      if (val.status === 'fuego') {
        activeCount++;
      } else if (val.status === 'quemado') {
        burnedCount++;
      }

      if (val.fuelId === 1) fuelDist.pastizal++;
      else if (val.fuelId === 4) fuelDist.matorral++;
      else if (val.fuelId === 8) fuelDist.bosqueRalo++;
      else if (val.fuelId === 10) fuelDist.bosqueDenso++;
    }

    this.activeFiresCount.set(activeCount);
    this.burnedCellsCount.set(burnedCount);
    this.fuelDistribution.set(fuelDist);
  }

  onIgnitionPointChange(point: [number, number]): void {
    this.ignitionPoint.set(point);
  }

  onSimulationSpeedChange(speed: number): void {
    this.simulationSpeed.set(speed);
    if (this.isRunning()) {
      this.simulation.setSpeed(speed);
    }
  }

  onStart(): void {
    const point = this.ignitionPoint();
    if (!point) return;
    this.burnedOrBurning.clear();
    this.cellStates.clear();
    this.maxRos.set(0.0);
    this.maxIntensity.set(0.0);
    this.maxFlameLength.set(0.0);
    this.burnedArea.set(0.0);
    this.suppressionActive.set(false);
    this.elapsedTime.set(0.0);
    this.activeFiresCount.set(0);
    this.burnedCellsCount.set(0);
    this.fuelDistribution.set({ pastizal: 0, matorral: 0, bosqueRalo: 0, bosqueDenso: 0 });
    this.showPerimeter.set(false);
    this.cellUpdates.set([]);

    this.simulation.startSimulation({
      windSpeed: this.windSpeed(),
      windDirection: this.windDirection(),
      ignitionPoint: point,
      humidity: this.humidity() / 100.0,
      temperature: this.temperature(),
      detectionTime: this.detectionTime(),
      simulationSpeed: this.simulationSpeed(),
    });
    this.isRunning.set(true);
    this.isPaused.set(false);
  }

  onPause(): void {
    this.simulation.pauseSimulation();
    this.isPaused.set(true);
  }

  onResume(): void {
    this.simulation.resumeSimulation();
    this.isRunning.set(true);
    this.isPaused.set(false);
  }

  onStop(): void {
    this.simulation.stopSimulation();
    this.isRunning.set(false);
    this.isPaused.set(false);
    this.burnedOrBurning.clear();
    this.cellStates.clear();
    this.maxRos.set(0.0);
    this.maxIntensity.set(0.0);
    this.maxFlameLength.set(0.0);
    this.burnedArea.set(0.0);
    this.suppressionActive.set(false);
    this.elapsedTime.set(0.0);
    this.activeFiresCount.set(0);
    this.burnedCellsCount.set(0);
    this.fuelDistribution.set({ pastizal: 0, matorral: 0, bosqueRalo: 0, bosqueDenso: 0 });
    this.showPerimeter.set(false);
  }

  onClear(): void {
    this.simulation.stopSimulation();
    this.isRunning.set(false);
    this.isPaused.set(false);
    this.ignitionPoint.set(null);

    this.maxRos.set(0.0);
    this.maxIntensity.set(0.0);
    this.maxFlameLength.set(0.0);
    this.burnedArea.set(0.0);
    this.elapsedTime.set(0.0);
    this.suppressionActive.set(false);
    this.activeFiresCount.set(0);
    this.burnedCellsCount.set(0);
    this.fuelDistribution.set({ pastizal: 0, matorral: 0, bosqueRalo: 0, bosqueDenso: 0 });
    this.showPerimeter.set(false);

    this.cellStates.clear();
    this.burnedOrBurning.clear();
    this.cellUpdates.set([]); // Limpia la visualización en el MapViewer
  }

  dismissError(): void {
    this.errorMessage.set(null);
  }
}
