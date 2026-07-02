import { Injectable, NgZone } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { Observable, timer, throwError, Subject } from 'rxjs';
import { map, retryWhen, mergeMap, tap } from 'rxjs/operators';
import {
  InitialData,
  SimulationConfig,
  CellUpdate,
  WsClientMessage,
  WsServerMessage,
  HealthResponse,
} from '../../../shared/interfaces';

const API_BASE = 'http://localhost:8001';
const WS_URL = 'ws://localhost:8001/api/simulation/ws';

const MAX_RETRIES = 10;
const INITIAL_RETRY_DELAY = 1000;
const MAX_RETRY_DELAY = 30000;

@Injectable({
  providedIn: 'root',
})
export class SimulationService {
  private wsSubject!: WebSocketSubject<WsServerMessage>;
  private connectionStatus$ = new Subject<'connected' | 'disconnected' | 'reconnecting'>();
  private currentConfig: SimulationConfig | null = null;

  constructor(private http: HttpClient, private ngZone: NgZone) {
    this.connect();
  }

  private connect(): void {
    this.wsSubject = webSocket<WsServerMessage>({
      url: WS_URL,
      openObserver: {
        next: () => {
          this.connectionStatus$.next('connected');
        },
      },
      closeObserver: {
        next: () => {
          this.connectionStatus$.next('disconnected');
        },
      },
    });
  }

  getInitialData(): Observable<InitialData> {
    return this.http.get<InitialData>(`${API_BASE}/api/simulation/initial`);
  }

  getHealth(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${API_BASE}/api/simulation/health`);
  }

  get rawMessages(): Observable<WsServerMessage> {
    return this.wsSubject.asObservable();
  }

  get cellUpdates(): Observable<CellUpdate[]> {
    return this.wsSubject.pipe(
      mergeMap((msg: WsServerMessage) => {
        if (Array.isArray(msg)) {
          return [msg];
        }
        return [];
      }),
      retryWhen((errors) =>
        errors.pipe(
          tap((err) => {
            this.connectionStatus$.next('reconnecting');
          }),
          mergeMap((err, attempt) => {
            if (attempt >= MAX_RETRIES) {
              return throwError(() => new Error('Max reconnection attempts reached'));
            }
            const delay = Math.min(INITIAL_RETRY_DELAY * Math.pow(2, attempt), MAX_RETRY_DELAY);
            return timer(delay);
          }),
          tap(() => {
            this.connect();
          })
        )
      )
    ) as Observable<CellUpdate[]>;
  }

  get connectionStatus(): Observable<'connected' | 'disconnected' | 'reconnecting'> {
    return this.connectionStatus$.asObservable();
  }

  startSimulation(config: SimulationConfig): void {
    this.currentConfig = config;
    this.send({ action: 'start', config });
  }

  pauseSimulation(): void {
    this.send({ action: 'pause' });
  }

  resumeSimulation(): void {
    this.send({ action: 'resume' });
  }

  stopSimulation(): void {
    this.currentConfig = null;
    this.send({ action: 'stop' });
  }

  setSpeed(speed: number): void {
    this.send({ action: 'set_speed', speed });
  }

  disconnect(): void {
    this.currentConfig = null;
    this.wsSubject.complete();
  }

  private send(msg: WsClientMessage): void {
    (this.wsSubject as WebSocketSubject<WsClientMessage | WsServerMessage>).next(msg);
  }
}
