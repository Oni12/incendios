import { Component, input, output, effect, inject, NgZone, OnDestroy } from '@angular/core';
import { CellUpdate, ZonePolygon } from '../../../../shared/interfaces';
import * as Cesium from 'cesium';

interface CellData {
  status: 'fuego' | 'quemado';
  intensity: number;
}

@Component({
  selector: 'app-map-viewer',
  imports: [],
  templateUrl: './map-viewer.html',
  styleUrl: './map-viewer.css',
})
export class MapViewer implements OnDestroy {
  zonePolygons = input<ZonePolygon[]>([]);
  cellUpdates = input<CellUpdate[]>([]);
  showPerimeter = input(false);
  isRunning = input(false);
  rows = input(100);
  cols = input(100);
  ignitionPointChange = output<[number, number]>();

  private viewer?: Cesium.Viewer;
  private zoneEntities: Cesium.Entity[] = [];
  private markerEntity?: Cesium.Entity;
  private perimeterLayerEntity?: Cesium.Entity;
  private handler?: Cesium.ScreenSpaceEventHandler;
  private cameraClampRemove?: () => void;

  private ngZone = inject(NgZone);

  private burnedCells = new Map<string, CellData>();
  private activeFires = new Map<string, CellData>();

  private pendingUpdates: CellUpdate[] = [];
  private rafId: number | null = null;
  private lastRenderTime = 0;
  private readonly TARGET_FPS = 30;
  private readonly FRAME_INTERVAL = 1000 / this.TARGET_FPS;

  private firePrimitive: Cesium.Primitive | null = null;
  private burnedPrimitive: Cesium.Primitive | null = null;

  private lastHullTime = 0;
  private readonly HULL_INTERVAL = 3000;

  private firePulseTimer: ReturnType<typeof setInterval> | null = null;
  private firePulsePhase = 0;
  private bloomStage: Cesium.PostProcessStage | null = null;
  private fireParticleSystems: Cesium.ParticleSystem[] = [];
  private lastParticleUpdate = 0;
  private readonly PARTICLE_UPDATE_INTERVAL = 1000;

  private minLat = 0;
  private maxLat = 0;
  private minLng = 0;
  private maxLng = 0;
  private latStep = 0;
  private lngStep = 0;

  private zoneBoundsComputed = false;
  private initialFlyDone = false;
  private readonly VIEW_PADDING_DEGREES = 0.015;

  constructor() {
    effect(() => {
      const polygons = this.zonePolygons();
      if (polygons.length && this.viewer) {
        this.activeFires.clear();
        this.burnedCells.clear();
        this.computeZoneBounds(polygons);
        if (this.perimeterLayerEntity) {
          this.viewer.entities.remove(this.perimeterLayerEntity);
          this.perimeterLayerEntity = undefined;
        }
        this.removeFirePrimitive();
        this.removeBurnedPrimitive();
        this.drawZones(polygons);
        this.constrainCameraToZone();
        this.initialFlyDone = true;
      }
    });

    effect(() => {
      const updates = this.cellUpdates();
      if (!this.viewer) return;
      if (updates.length === 0 && !this.isRunning()) {
        this.clearAll();
        return;
      }
      if (updates.length > 0) {
        this.pendingUpdates.push(...updates);
        this.scheduleRender();
      }
    });

    effect(() => {
      const show = this.showPerimeter();
      const _ = this.cellUpdates();
      if (this.viewer) {
        this.updatePerimeter(show);
      }
    });
  }

  ngAfterViewInit(): void {
    this.initMap();
  }

  ngOnDestroy(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
    }
    this.stopFirePulse();
    this.removeBloomEffect();
    this.removeAllParticles();
    if (this.handler) {
      this.handler.destroy();
    }
    if (this.cameraClampRemove) {
      this.cameraClampRemove();
    }
    if (this.viewer) {
      this.viewer.destroy();
    }
  }

  private computeZoneBounds(polygons: ZonePolygon[]): void {
    let minLat = Infinity;
    let maxLat = -Infinity;
    let minLng = Infinity;
    let maxLng = -Infinity;

    for (const poly of polygons) {
      for (const coord of poly.coordinates) {
        const lat = coord[0];
        const lng = coord[1];
        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
        if (lng < minLng) minLng = lng;
        if (lng > maxLng) maxLng = lng;
      }
    }

    this.minLat = minLat;
    this.maxLat = maxLat;
    this.minLng = minLng;
    this.maxLng = maxLng;
    this.latStep = (maxLat - minLat) / this.rows();
    this.lngStep = (maxLng - minLng) / this.cols();
    this.zoneBoundsComputed = true;
  }

  private scheduleRender(): void {
    if (this.rafId !== null) return;
    this.rafId = requestAnimationFrame((time) => {
      this.rafId = null;
      const elapsed = time - this.lastRenderTime;
      if (elapsed < this.FRAME_INTERVAL) {
        this.scheduleRender();
        return;
      }
      this.lastRenderTime = time;
      this.flushRender();
    });
  }

  private fireInstanceIds: Map<string, string> = new Map();
  private burnedInstanceIds: Map<string, string> = new Map();
  private lastFireCount = 0;
  private lastBurnedCount = 0;

  private flushRender(): void {
    const updates = this.pendingUpdates;
    if (updates.length === 0) return;
    this.pendingUpdates = [];
    let fireGeometryChanged = false;
    let burnedGeometryChanged = false;

    for (const update of updates) {
      const key = `${update.row},${update.col}`;
      const cellData: CellData = {
        status: update.status as 'fuego' | 'quemado',
        intensity: update.intensity || 0,
      };

      if (update.status === 'fuego') {
        fireGeometryChanged = fireGeometryChanged || !this.activeFires.has(key);
        burnedGeometryChanged = burnedGeometryChanged || this.burnedCells.has(key);
        this.activeFires.set(key, cellData);
        this.burnedCells.delete(key);
      } else if (update.status === 'quemado') {
        burnedGeometryChanged = burnedGeometryChanged || !this.burnedCells.has(key);
        fireGeometryChanged = fireGeometryChanged || this.activeFires.has(key);
        this.burnedCells.set(key, cellData);
        this.activeFires.delete(key);
      }
    }

    const fireCountChanged = Math.abs(this.activeFires.size - this.lastFireCount) > this.lastFireCount * 0.2 || this.activeFires.size < 50;
    const burnedCountChanged = Math.abs(this.burnedCells.size - this.lastBurnedCount) > this.lastBurnedCount * 0.2 || this.burnedCells.size < 50;

    if (fireGeometryChanged || fireCountChanged || this.activeFires.size === 0) {
      this.rebuildFirePrimitive();
    } else {
      this.updateFirePrimitiveColors();
    }

    if (burnedGeometryChanged || burnedCountChanged || this.burnedCells.size === 0) {
      this.rebuildBurnedPrimitive();
    } else {
      this.updateBurnedPrimitiveColors();
    }

    this.lastFireCount = this.activeFires.size;
    this.lastBurnedCount = this.burnedCells.size;
    this.updateFireParticles();

    const now = Date.now();
    if (now - this.lastHullTime > this.HULL_INTERVAL) {
      this.lastHullTime = now;
      const show = this.showPerimeter();
      if (show) {
        this.updatePerimeter(true);
      }
    }

    this.viewer!.scene.requestRender();
  }

  private updateFirePrimitiveColors(): void {
    if (this.activeFires.size === 0 || !this.firePrimitive) return;

    const maxIntensity = Math.max(...Array.from(this.activeFires.values()).map(f => f.intensity), 1);
    const lut = this.getFireColorLUT();

    for (const [key, data] of this.activeFires) {
      const instId = this.fireInstanceIds.get(key);
      if (instId == null) continue;

      const attrs = this.firePrimitive!.getGeometryInstanceAttributes(instId);
      if (!attrs) continue;

      const intensityRatio = Math.min(data.intensity / maxIntensity, 1.0);
      const ci = Math.min(255, Math.max(0, Math.round(intensityRatio * 255)));

      attrs.color = Cesium.ColorGeometryInstanceAttribute.toValue(
        new Cesium.Color(lut[ci * 4] / 255, lut[ci * 4 + 1] / 255, lut[ci * 4 + 2] / 255, lut[ci * 4 + 3] / 255)
      );
    }
  }

  private updateBurnedPrimitiveColors(): void {
    if (!this.burnedPrimitive || this.burnedCells.size === 0) return;
  }

  private rebuildFirePrimitive(): void {
    this.removeFirePrimitive();
    if (this.activeFires.size === 0) return;

    const maxIntensity = Math.max(...Array.from(this.activeFires.values()).map(f => f.intensity), 1);
    const lut = this.getFireColorLUT();

    const instances: Cesium.GeometryInstance[] = [];
    const colors: Cesium.Color[] = [];
    let idx = 0;

    for (const [key, data] of this.activeFires) {
      const [r, c] = key.split(',').map(Number);
      const cellMaxLat = this.maxLat - r * this.latStep;
      const cellMinLat = this.maxLat - (r + 1) * this.latStep;
      const cellMinLng = this.minLng + c * this.lngStep;
      const cellMaxLng = this.minLng + (c + 1) * this.lngStep;

      const intensityRatio = Math.min(data.intensity / maxIntensity, 1.0);
      const ci = Math.min(255, Math.max(0, Math.round(intensityRatio * 255)));
      const color = new Cesium.Color(lut[ci * 4] / 255, lut[ci * 4 + 1] / 255, lut[ci * 4 + 2] / 255, lut[ci * 4 + 3] / 255);

      instances.push(new Cesium.GeometryInstance({
        id: `fire_${idx}`,
        geometry: new Cesium.RectangleGeometry({
          rectangle: Cesium.Rectangle.fromDegrees(cellMinLng, cellMinLat, cellMaxLng, cellMaxLat),
        }),
        attributes: {
          color: Cesium.ColorGeometryInstanceAttribute.fromColor(color),
        },
      }));
      colors.push(color);
      this.fireInstanceIds.set(key, `fire_${idx}`);
      idx++;
    }

    this.firePrimitive = this.viewer!.scene.primitives.add(
      new Cesium.Primitive({
        geometryInstances: instances,
        appearance: new Cesium.PerInstanceColorAppearance({
          flat: true,
          translucent: true,
        }),
        asynchronous: false,
      })
    );
  }

  private static _fireColorLUT: Uint8Array | null = null;

  private getFireColorLUT(): Uint8Array {
    if (MapViewer._fireColorLUT) return MapViewer._fireColorLUT;
    const lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      let r: number, g: number, b: number, a: number;
      if (t > 0.7) {
        const s = (t - 0.7) / 0.3;
        r = Math.round(255 * (1 - s * 0.0));
        g = Math.round(68 * (1 - s * 0.5));
        b = Math.round(0 + s * 255);
        a = Math.round((0.8 + s * 0.1) * 255);
      } else if (t > 0.3) {
        const s = (t - 0.3) / 0.4;
        r = 255;
        g = Math.round(170 - s * 102);
        b = 0;
        a = Math.round((0.7 + s * 0.1) * 255);
      } else {
        const s = t / 0.3;
        r = 255;
        g = Math.round(204 - s * 34);
        b = 0;
        a = Math.round((0.6 + s * 0.1) * 255);
      }
      lut[i * 4] = r;
      lut[i * 4 + 1] = g;
      lut[i * 4 + 2] = b;
      lut[i * 4 + 3] = a;
    }
    MapViewer._fireColorLUT = lut;
    return lut;
  }

  private rebuildBurnedPrimitive(): void {
    this.removeBurnedPrimitive();
    if (this.burnedCells.size === 0) return;

    const burnedColor = Cesium.Color.fromCssColorString('#facc15').withAlpha(0.45);
    const instances: Cesium.GeometryInstance[] = [];
    let idx = 0;

    for (const key of this.burnedCells.keys()) {
      const [r, c] = key.split(',').map(Number);
      const cellMaxLat = this.maxLat - r * this.latStep;
      const cellMinLat = this.maxLat - (r + 1) * this.latStep;
      const cellMinLng = this.minLng + c * this.lngStep;
      const cellMaxLng = this.minLng + (c + 1) * this.lngStep;

      instances.push(new Cesium.GeometryInstance({
        id: `burned_${idx}`,
        geometry: new Cesium.RectangleGeometry({
          rectangle: Cesium.Rectangle.fromDegrees(cellMinLng, cellMinLat, cellMaxLng, cellMaxLat),
        }),
        attributes: {
          color: Cesium.ColorGeometryInstanceAttribute.fromColor(burnedColor),
        },
      }));
      this.burnedInstanceIds.set(key, `burned_${idx}`);
      idx++;
    }

    this.burnedPrimitive = this.viewer!.scene.primitives.add(
      new Cesium.Primitive({
        geometryInstances: instances,
        appearance: new Cesium.PerInstanceColorAppearance({
          flat: true,
          translucent: true,
        }),
        asynchronous: false,
      })
    );
  }

  private removeFirePrimitive(): void {
    if (this.firePrimitive && this.viewer) {
      this.viewer.scene.primitives.remove(this.firePrimitive);
      this.firePrimitive = null;
    }
    this.fireInstanceIds.clear();
  }

  private removeBurnedPrimitive(): void {
    if (this.burnedPrimitive && this.viewer) {
      this.viewer.scene.primitives.remove(this.burnedPrimitive);
      this.burnedPrimitive = null;
    }
    this.burnedInstanceIds.clear();
  }

  private async initMap(): Promise<void> {
    (window as any).CESIUM_BASE_URL = '/assets/cesium/';

    const esriImagery = new Cesium.UrlTemplateImageryProvider({
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      maximumLevel: 19,
      credit: 'Esri, Maxar, Earthstar Geographics',
    });

    this.viewer = new Cesium.Viewer('map', {
      baseLayer: new Cesium.ImageryLayer(esriImagery),
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
      navigationHelpButton: false,
      requestRenderMode: true,
      sceneModePicker: false,
      selectionIndicator: false,
      timeline: false,
      animation: false,
      fullscreenButton: false,
    });

    this.viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(-64.85, -21.55, 50000),
      orientation: {
        heading: 0,
        pitch: Cesium.Math.toRadians(-35),
        roll: 0,
      },
    });

    this.viewer.scene.globe.depthTestAgainstTerrain = false;
    this.viewer.scene.screenSpaceCameraController.minimumZoomDistance = 250;
    this.viewer.scene.screenSpaceCameraController.maximumZoomDistance = 90000;

    this.setupBloomEffect();
    this.startFirePulse();

    this.handler = new Cesium.ScreenSpaceEventHandler(this.viewer.scene.canvas);
    this.handler.setInputAction((click: any) => {
      const ray = this.viewer!.camera.getPickRay(click.position);
      if (!ray) return;
      const cartesian = this.viewer!.scene.globe.pick(ray, this.viewer!.scene);
      if (cartesian) {
        const cartographic = Cesium.Cartographic.fromCartesian(cartesian);
        const lat = Cesium.Math.toDegrees(cartographic.latitude);
        const lng = Cesium.Math.toDegrees(cartographic.longitude);
        const [ignitionLat, ignitionLng] = this.snapToGridCenter(lat, lng);

        this.ngZone.run(() => {
          this.placeIgnitionMarker(ignitionLat, ignitionLng);
          this.ignitionPointChange.emit([ignitionLat, ignitionLng]);
        });
      }
    }, Cesium.ScreenSpaceEventType.LEFT_CLICK);

    try {
      const kmlDataSource = await Cesium.KmlDataSource.load('/Sama.kml', {
        camera: this.viewer.camera,
        canvas: this.viewer.scene.canvas,
        clampToGround: true,
      });
      this.viewer.dataSources.add(kmlDataSource);

      kmlDataSource.entities.values.forEach((entity) => {
        if (entity.polygon) {
          entity.polygon.fill = new Cesium.ConstantProperty(false);
          entity.polygon.outline = new Cesium.ConstantProperty(true);
        }
      });
    } catch (e) {
      console.warn('Error al cargar Sama.kml:', e);
    }

    const polygons = this.zonePolygons();
    if (polygons.length) {
      this.computeZoneBounds(polygons);
      this.drawZones(polygons);
      this.constrainCameraToZone();
      this.initialFlyDone = true;
    }

    setTimeout(() => {
      if (!this.initialFlyDone && this.viewer) {
        const deferredPolygons = this.zonePolygons();
        if (deferredPolygons.length) {
          this.computeZoneBounds(deferredPolygons);
          this.drawZones(deferredPolygons);
          this.constrainCameraToZone();
          this.initialFlyDone = true;
        }
      }
    }, 1500);
  }

  private drawZones(polygons: ZonePolygon[]): void {
    if (!this.viewer) return;

    this.zoneEntities.forEach((entity) => this.viewer!.entities.remove(entity));
    this.zoneEntities = [];

    const allPositions: Cesium.Cartesian3[] = [];

    for (const poly of polygons) {
      let color = '#e63946';
      if (poly.year === 2024) {
        color = '#22c55e';
      } else if (poly.year === 2025) {
        color = '#3b82f6';
      }

      const positions = poly.coordinates.map((coord) => Cesium.Cartesian3.fromDegrees(coord[1], coord[0]));
      allPositions.push(...positions);

      const borderPositions = [...positions];
      if (borderPositions.length > 0) {
        borderPositions.push(borderPositions[0]);
      }

      const polylineEntity = this.viewer.entities.add({
        polyline: {
          positions: borderPositions,
          width: 3.0,
          material: Cesium.Color.fromCssColorString(color),
          clampToGround: true,
        },
      });
      this.zoneEntities.push(polylineEntity);
    }

    if (allPositions.length > 0) {
      const boundingSphere = Cesium.BoundingSphere.fromPoints(allPositions);
      this.viewer.camera.flyToBoundingSphere(boundingSphere, {
        duration: 2.0,
        offset: new Cesium.HeadingPitchRange(
          Cesium.Math.toRadians(0),
          Cesium.Math.toRadians(-35),
          boundingSphere.radius * 2.2
        ),
      });
    }
  }

  private getPaddedZoneBounds(): { minLat: number; maxLat: number; minLng: number; maxLng: number } {
    return {
      minLat: this.minLat - this.VIEW_PADDING_DEGREES,
      maxLat: this.maxLat + this.VIEW_PADDING_DEGREES,
      minLng: this.minLng - this.VIEW_PADDING_DEGREES,
      maxLng: this.maxLng + this.VIEW_PADDING_DEGREES,
    };
  }

  private constrainCameraToZone(): void {
    if (!this.viewer || !this.zoneBoundsComputed || this.cameraClampRemove) return;

    let clamping = false;
    this.cameraClampRemove = this.viewer.camera.changed.addEventListener(() => {
      if (!this.viewer || clamping) return;

      const bounds = this.getPaddedZoneBounds();
      const cartographic = this.viewer.camera.positionCartographic;
      const lat = Cesium.Math.toDegrees(cartographic.latitude);
      const lng = Cesium.Math.toDegrees(cartographic.longitude);
      const clampedLat = Math.max(bounds.minLat, Math.min(bounds.maxLat, lat));
      const clampedLng = Math.max(bounds.minLng, Math.min(bounds.maxLng, lng));

      if (clampedLat === lat && clampedLng === lng) return;

      clamping = true;
      this.viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(clampedLng, clampedLat, cartographic.height),
        orientation: {
          heading: this.viewer.camera.heading,
          pitch: this.viewer.camera.pitch,
          roll: this.viewer.camera.roll,
        },
      });
      clamping = false;
    });
  }

  private placeIgnitionMarker(lat: number, lng: number): void {
    if (!this.viewer) return;

    if (this.markerEntity) {
      this.viewer.entities.remove(this.markerEntity);
    }

    this.markerEntity = this.viewer.entities.add({
      position: Cesium.Cartesian3.fromDegrees(lng, lat),
      label: {
        text: '🔥',
        font: '28px sans-serif',
        horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
        verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND,
      },
    });
  }

  private snapToGridCenter(lat: number, lng: number): [number, number] {
    if (!this.zoneBoundsComputed || this.latStep === 0 || this.lngStep === 0) {
      return [lat, lng];
    }

    const row = Math.max(0, Math.min(
      Math.floor(((this.maxLat - lat) / (this.maxLat - this.minLat)) * this.rows()),
      this.rows() - 1
    ));
    const col = Math.max(0, Math.min(
      Math.floor(((lng - this.minLng) / (this.maxLng - this.minLng)) * this.cols()),
      this.cols() - 1
    ));

    return [
      this.maxLat - (row + 0.5) * this.latStep,
      this.minLng + (col + 0.5) * this.lngStep,
    ];
  }

  clearAll(): void {
    this.activeFires.clear();
    this.burnedCells.clear();
    this.fireInstanceIds.clear();
    this.burnedInstanceIds.clear();
    this.lastFireCount = 0;
    this.lastBurnedCount = 0;
    this.removeFirePrimitive();
    this.removeBurnedPrimitive();
    this.removeAllParticles();
    this.pendingUpdates = [];
    if (this.viewer) {
      if (this.perimeterLayerEntity) {
        this.viewer.entities.remove(this.perimeterLayerEntity);
        this.perimeterLayerEntity = undefined;
      }
      if (this.markerEntity) {
        this.viewer.entities.remove(this.markerEntity);
        this.markerEntity = undefined;
      }
    }
    this.lastHullTime = 0;
  }

  private setupBloomEffect(): void {
    if (!this.viewer) return;
    const fs = `
      uniform sampler2D colorTexture;
      in vec2 v_textureCoordinates;
      void main() {
        vec4 color = texture(colorTexture, v_textureCoordinates);
        float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
        vec3 bloom = color.rgb * smoothstep(0.6, 1.0, brightness) * 0.4;
        out_FragColor = vec4(color.rgb + bloom, color.a);
      }
    `;
    this.bloomStage = new Cesium.PostProcessStage({
      fragmentShader: fs,
    });
    this.viewer.scene.postProcessStages.add(this.bloomStage);
  }

  private removeBloomEffect(): void {
    if (this.bloomStage && this.viewer) {
      this.viewer.scene.postProcessStages.remove(this.bloomStage);
      this.bloomStage = null;
    }
  }

  private startFirePulse(): void {
    this.firePulseTimer = setInterval(() => {
      this.firePulsePhase = (this.firePulsePhase + 0.15) % (Math.PI * 2);
      this.updateFirePulse();
    }, 100);
  }

  private stopFirePulse(): void {
    if (this.firePulseTimer) {
      clearInterval(this.firePulseTimer);
      this.firePulseTimer = null;
    }
  }

  private updateFirePulse(): void {
    if (!this.firePrimitive || this.activeFires.size === 0) return;

    const pulse = 0.55 + 0.2 * Math.sin(this.firePulsePhase);
    const heatPulse = 0.65 + 0.15 * Math.sin(this.firePulsePhase + 1.0);

    try {
      const colorAttr = this.firePrimitive.getGeometryInstanceAttributes('0');
      if (colorAttr && colorAttr.color) {
        const baseColor = Cesium.Color.fromCssColorString('#ef4444');
        const hotColor = Cesium.Color.fromCssColorString('#ff6b35');
        const blended = Cesium.Color.lerp(hotColor, baseColor, pulse, new Cesium.Color());
        blended.alpha = heatPulse;
        colorAttr.color = Cesium.ColorGeometryInstanceAttribute.fromColor(blended).value;
      }
    } catch (_) {
    }
  }

  private updateFireParticles(): void {
    if (!this.viewer || this.activeFires.size === 0) {
      this.removeAllParticles();
      return;
    }

    const now = Date.now();
    if (now - this.lastParticleUpdate < this.PARTICLE_UPDATE_INTERVAL) return;
    this.lastParticleUpdate = now;

    this.removeAllParticles();

    const sortedFires = Array.from(this.activeFires.entries())
      .sort((a, b) => b[1].intensity - a[1].intensity)
      .slice(0, 8);

    for (const [key, data] of sortedFires) {
      const [r, c] = key.split(',').map(Number);
      const lat = this.maxLat - (r + 0.5) * this.latStep;
      const lng = this.minLng + (c + 0.5) * this.lngStep;

      const particleSystem = this.viewer.scene.primitives.add(
        new Cesium.ParticleSystem({
          image: this.createFireParticleImage(),
          emitter: new Cesium.CircleEmitter(0.00005),
          emissionRate: 15 + Math.min(data.intensity / 100, 20),
          minimumParticleLife: 0.5,
          maximumParticleLife: 1.5,
          minimumSpeed: 0.5,
          maximumSpeed: 2.0,
          startScale: 2.0,
          endScale: 6.0,
          startColor: Cesium.Color.fromCssColorString('#ffaa00').withAlpha(0.8),
          endColor: Cesium.Color.fromCssColorString('#ff2200').withAlpha(0.0),
          lifetime: 16.0,
          sizeInMeters: false,
        })
      );

      particleSystem.setPosition(
        Cesium.Cartesian3.fromDegrees(lng, lat, 10.0)
      );
      this.fireParticleSystems.push(particleSystem);
    }
  }

  private static _cachedFireImage: HTMLCanvasElement | null = null;

  private createFireParticleImage(): HTMLCanvasElement {
    if (MapViewer._cachedFireImage) return MapViewer._cachedFireImage;
    const canvas = document.createElement('canvas');
    canvas.width = 16;
    canvas.height = 16;
    const ctx = canvas.getContext('2d')!;
    const gradient = ctx.createRadialGradient(8, 8, 0, 8, 8, 8);
    gradient.addColorStop(0, 'rgba(255, 255, 200, 1.0)');
    gradient.addColorStop(0.3, 'rgba(255, 170, 0, 0.9)');
    gradient.addColorStop(0.6, 'rgba(255, 80, 0, 0.5)');
    gradient.addColorStop(1.0, 'rgba(255, 30, 0, 0.0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 16, 16);
    MapViewer._cachedFireImage = canvas;
    return canvas;
  }

  private removeAllParticles(): void {
    if (this.viewer) {
      for (const ps of this.fireParticleSystems) {
        this.viewer.scene.primitives.remove(ps);
      }
    }
    this.fireParticleSystems = [];
  }

  private updatePerimeter(show: boolean): void {
    if (this.perimeterLayerEntity) {
      this.viewer?.entities.remove(this.perimeterLayerEntity);
      this.perimeterLayerEntity = undefined;
    }

    if (!show || !this.viewer) {
      if (this.viewer) this.viewer.scene.requestRender();
      return;
    }

    if (!this.zoneBoundsComputed) return;

    const allCells = [...this.burnedCells.keys(), ...this.activeFires.keys()];
    if (allCells.length === 0) return;

    const pts: { x: number; y: number }[] = [];
    for (const key of allCells) {
      const [r, c] = key.split(',').map(Number);
      const cellMaxLat = this.maxLat - r * this.latStep;
      const cellMinLat = this.maxLat - (r + 1) * this.latStep;
      const cellMinLng = this.minLng + c * this.lngStep;
      const cellMaxLng = this.minLng + (c + 1) * this.lngStep;

      pts.push({ x: cellMinLng, y: cellMinLat });
      pts.push({ x: cellMaxLng, y: cellMinLat });
      pts.push({ x: cellMinLng, y: cellMaxLat });
      pts.push({ x: cellMaxLng, y: cellMaxLat });
    }

    const hullPoints = this.getConvexHull(pts);
    if (hullPoints.length >= 3) {
      const positions = hullPoints.map((pt) => Cesium.Cartesian3.fromDegrees(pt.x, pt.y));
      positions.push(positions[0]);

      this.perimeterLayerEntity = this.viewer.entities.add({
        polyline: {
          positions: positions,
          width: 3.0,
          material: new Cesium.PolylineDashMaterialProperty({
            color: Cesium.Color.fromCssColorString('#ef4444'),
            dashLength: 16.0,
          }),
          clampToGround: true,
        },
      });
      this.viewer!.scene.requestRender();
    }
  }

  private getConvexHull(pointsList: { x: number; y: number }[]): { x: number; y: number }[] {
    const seen = new Set<string>();
    const uniquePts: { x: number; y: number }[] = [];
    for (const p of pointsList) {
      const k = `${p.x.toFixed(6)},${p.y.toFixed(6)}`;
      if (!seen.has(k)) {
        seen.add(k);
        uniquePts.push(p);
      }
    }

    if (uniquePts.length <= 3) return uniquePts;

    let pivot = uniquePts[0];
    for (let i = 1; i < uniquePts.length; i++) {
      if (uniquePts[i].y < pivot.y || (uniquePts[i].y === pivot.y && uniquePts[i].x < pivot.x)) {
        pivot = uniquePts[i];
      }
    }

    const crossProduct = (p1: any, p2: any, p3: any) => {
      return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
    };

    const distSq = (p1: any, p2: any) => {
      return (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2;
    };

    const candidates = uniquePts.filter((p) => p !== pivot);
    candidates.sort((a, b) => {
      const order = crossProduct(pivot, a, b);
      if (order === 0) {
        return distSq(pivot, a) < distSq(pivot, b) ? -1 : 1;
      }
      return order > 0 ? -1 : 1;
    });

    const hull = [pivot, candidates[0]];
    for (let i = 1; i < candidates.length; i++) {
      while (hull.length >= 2 && crossProduct(hull[hull.length - 2], hull[hull.length - 1], candidates[i]) <= 0) {
        hull.pop();
      }
      hull.push(candidates[i]);
    }
    return hull;
  }
}
