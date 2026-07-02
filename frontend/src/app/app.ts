// src/app/app.ts
import { Component } from '@angular/core';
import { SimulatorComponent } from './features/simulator/simulator';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [SimulatorComponent], 
  template: `<app-simulator></app-simulator>`,
  styles: []
})
export class App {
  title = 'frontend';
}