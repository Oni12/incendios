import { Component, input, output, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';

export interface FuelDist {
  pastizal: number;
  matorral: number;
  bosqueRalo: number;
  bosqueDenso: number;
}

@Component({
  selector: 'app-stats-panel',
  standalone: true,
  imports: [DecimalPipe],
  templateUrl: './stats-panel.html',
  styleUrl: './stats-panel.css'
})
export class StatsPanel {
  elapsedTime = input(0.0);
  burnedArea = input(0.0);
  maxRos = input(0.0);
  suppressionActive = input(false);
  
  // Nuevas estadísticas avanzadas
  maxIntensity = input(0.0);
  maxFlameLength = input(0.0);
  activeFiresCount = input(0);
  burnedCellsCount = input(0);
  fuelDistribution = input<FuelDist>({
    pastizal: 0,
    matorral: 0,
    bosqueRalo: 0,
    bosqueDenso: 0
  });

  perimeterToggle = output<boolean>();
  showPerimeter = signal(false);

  togglePerimeter() {
    const nextState = !this.showPerimeter();
    this.showPerimeter.set(nextState);
    this.perimeterToggle.emit(nextState);
  }

  getPercent(count: number): number {
    const total = this.fuelDistribution().pastizal + 
                  this.fuelDistribution().matorral + 
                  this.fuelDistribution().bosqueRalo + 
                  this.fuelDistribution().bosqueDenso;
    return total > 0 ? (count / total) * 100 : 0;
  }
}
