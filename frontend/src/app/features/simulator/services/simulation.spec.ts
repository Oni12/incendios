import { TestBed } from '@angular/core/testing';

import { Simulation } from './simulation';

describe('Simulation', () => {
  let service: Simulation;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Simulation);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
