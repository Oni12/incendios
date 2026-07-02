import { Component, input, output } from '@angular/core';

@Component({
  selector: 'app-control-panel',
  imports: [],
  templateUrl: './control-panel.html',
  styleUrl: './control-panel.css',
})
export class ControlPanel {
  windSpeed = input(0);
  windDirection = input(0);
  humidity = input(0);
  temperature = input(0);
  detectionTime = input(0);
  simulationSpeed = input(1);
  canStart = input(false);
  isRunning = input(false);
  isPaused = input(false);

  // Estadísticas calculadas recibidas del componente principal
  maxRos = input(0.0);
  burnedArea = input(0.0);
  suppressionActive = input(false);
  elapsedTime = input(0.0);

  windSpeedChange = output<number>();
  windDirectionChange = output<number>();
  humidityChange = output<number>();
  temperatureChange = output<number>();
  detectionTimeChange = output<number>();
  simulationSpeedChange = output<number>();
  startSimulation = output<void>();
  pauseSimulation = output<void>();
  resumeSimulation = output<void>();
  stopSimulation = output<void>();
  clearSimulation = output<void>();

  onSpeedInput(event: Event) {
    const val = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(val)) this.windSpeedChange.emit(val);
  }

  onDirectionInput(event: Event) {
    const val = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(val)) this.windDirectionChange.emit(val);
  }

  onHumidityInput(event: Event) {
    const val = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(val)) this.humidityChange.emit(val);
  }

  onTemperatureInput(event: Event) {
    const val = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(val)) this.temperatureChange.emit(val);
  }

  onDetectionTimeInput(event: Event) {
    const val = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(val)) this.detectionTimeChange.emit(val);
  }

  onSimulationSpeedInput(event: Event) {
    const val = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(val)) this.simulationSpeedChange.emit(val);
  }
}
