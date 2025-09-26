/* Interactive 3D visualisations for the Lagrange multipliers article. */
(function () {
  const gridSize = 45;
  const domain = { min: -1.5, max: 1.5 };

  function linspace(min, max, count) {
    if (count === 1) return [min];
    const step = (max - min) / (count - 1);
    return Array.from({ length: count }, (_, idx) => min + step * idx);
  }

  function buildSurfaceData() {
    const xVals = linspace(domain.min, domain.max, gridSize);
    const yVals = linspace(domain.min, domain.max, gridSize);
    const zMatrix = yVals.map((y) => xVals.map((x) => x * y));
    return { xVals, yVals, zMatrix };
  }

  function makeSurfaceTrace(xVals, yVals, zMatrix) {
    return {
      type: 'surface',
      name: 'z = xy',
      showscale: false,
      x: xVals,
      y: yVals,
      z: zMatrix,
      opacity: 0.92,
      colorscale: 'Viridis',
    };
  }

  function makeConstraintTrace(samples) {
    return {
      type: 'scatter3d',
      mode: 'lines',
      name: 'Constraint x² + y² = 1',
      showlegend: true,
      line: { color: '#1f77b4', width: 6 },
      x: samples.map((p) => p.x),
      y: samples.map((p) => p.y),
      z: samples.map((p) => p.z),
      hovertemplate: 'Constraint path:<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>',
    };
  }

  function circleConstraintSamples(sampleCount) {
    return Array.from({ length: sampleCount }, (_, idx) => {
      const t = (idx / (sampleCount - 1)) * Math.PI * 2;
      const x = Math.cos(t);
      const y = Math.sin(t);
      return { x, y, z: x * y };
    });
  }

  function makeGradientTrace(point, color, name) {
    const gradStep = 0.45;
    const x2 = point.x + point.grad[0] * gradStep;
    const y2 = point.y + point.grad[1] * gradStep;
    const z2 = x2 * y2;
    return {
      type: 'scatter3d',
      mode: 'lines',
      name,
      showlegend: true,
      line: { color, width: 6 },
      x: [point.x, x2],
      y: [point.y, y2],
      z: [point.z, z2],
      hovertemplate: `${name}:<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>`,
    };
  }

  function makePointTrace(point, color, label) {
    return {
      type: 'scatter3d',
      mode: 'markers+text',
      name: label,
      showlegend: true,
      marker: { size: 7, color, symbol: 'circle' },
      text: [label],
      textposition: 'top center',
      x: [point.x],
      y: [point.y],
      z: [point.z],
      hovertemplate: `${label}:<br>x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>`,
    };
  }

  function makePlaneMesh(zValue, name, color) {
    const corners = [
      { x: domain.min, y: domain.min },
      { x: domain.max, y: domain.min },
      { x: domain.max, y: domain.max },
      { x: domain.min, y: domain.max },
    ];
    return {
      type: 'mesh3d',
      name,
      showlegend: true,
      hoverinfo: 'skip',
      opacity: 0.35,
      color,
      x: corners.map((c) => c.x),
      y: corners.map((c) => c.y),
      z: corners.map(() => zValue),
      i: [0, 0],
      j: [1, 2],
      k: [2, 3],
    };
  }

  function ensurePlotly(callback) {
    if (window.Plotly) {
      callback();
      return;
    }
    console.warn('Plotly not available; interactive plots skipped.');
  }

  function renderPlots() {
    const intersectionContainer = document.getElementById('lagrange-intersection-3d');
    const tangencyContainer = document.getElementById('lagrange-tangency-3d');
    if (!intersectionContainer || !tangencyContainer) {
      return;
    }

    const { xVals, yVals, zMatrix } = buildSurfaceData();
    const constraintSamples = circleConstraintSamples(160);
    const surfaceTraceIntersection = makeSurfaceTrace(xVals, yVals, zMatrix);
    const surfaceTraceTangency = makeSurfaceTrace(xVals, yVals, zMatrix);
    const constraintTraceIntersection = makeConstraintTrace(constraintSamples);
    const constraintTraceTangency = makeConstraintTrace(constraintSamples);

    const intersectionPoint = { x: 1, y: 0, z: 0, grad: [0, 1], gradConstraint: [2, 0] };
    const tangencyPoint = {
      x: Math.SQRT1_2,
      y: Math.SQRT1_2,
      z: 0.5,
      grad: [Math.SQRT1_2, Math.SQRT1_2],
      gradConstraint: [Math.SQRT2, Math.SQRT2],
    };

    const intersectionData = [
      surfaceTraceIntersection,
      constraintTraceIntersection,
      makePlaneMesh(0, 'Level set f=0', 'rgba(255, 127, 14, 0.6)'),
      makePointTrace(intersectionPoint, '#d62728', 'Intersection point'),
      makeGradientTrace(intersectionPoint, '#d62728', '∇f at (1,0)'),
      makeGradientTrace(
        { ...intersectionPoint, grad: intersectionPoint.gradConstraint },
        '#2ca02c',
        '∇g at (1,0)'
      ),
    ];

    const tangencyData = [
      surfaceTraceTangency,
      constraintTraceTangency,
      makePlaneMesh(0.5, 'Level set f=0.5', 'rgba(214, 39, 40, 0.6)'),
      makePointTrace(tangencyPoint, '#9467bd', 'Tangency point'),
      makeGradientTrace(tangencyPoint, '#d62728', '∇f at optimum'),
      makeGradientTrace(
        { ...tangencyPoint, grad: tangencyPoint.gradConstraint },
        '#2ca02c',
        '∇g at optimum'
      ),
    ];

    const baseScene = {
      xaxis: { title: 'x', zeroline: false, range: [domain.min, domain.max] },
      yaxis: { title: 'y', zeroline: false, range: [domain.min, domain.max] },
      zaxis: { title: 'f(x,y) = xy', range: [-0.8, 0.8] },
      aspectmode: 'cube',
      camera: { eye: { x: 1.4, y: 1.2, z: 0.9 } },
    };

    Plotly.newPlot(
      intersectionContainer,
      intersectionData,
      {
        title: 'Intersection: Level set crosses the constraint',
        scene: baseScene,
        legend: { orientation: 'h' },
        margin: { l: 0, r: 0, t: 50, b: 0 },
        height: 520,
      },
      { responsive: true }
    );

    Plotly.newPlot(
      tangencyContainer,
      tangencyData,
      {
        title: 'Tangency: Gradients align along the constraint',
        scene: baseScene,
        legend: { orientation: 'h' },
        margin: { l: 0, r: 0, t: 50, b: 0 },
        height: 520,
      },
      { responsive: true }
    );
  }

  document.addEventListener('DOMContentLoaded', function () {
    ensurePlotly(renderPlots);
  });
})();
