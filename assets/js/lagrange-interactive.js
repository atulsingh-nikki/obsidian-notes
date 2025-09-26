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

  function circle2DTrace(samples) {
    return {
      type: 'scatter',
      mode: 'lines',
      name: 'Constraint x² + y² = 1',
      line: { color: '#1f77b4', width: 3 },
      x: samples.map((p) => p.x),
      y: samples.map((p) => p.y),
      hovertemplate: 'Constraint:<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>',
    };
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

  function make2DArrowTrace(point, grad, color, label) {
    const gradStep = 0.55;
    const x2 = point.x + grad[0] * gradStep;
    const y2 = point.y + grad[1] * gradStep;
    return {
      type: 'scatter',
      mode: 'lines+markers',
      name: label,
      line: { color, width: 3 },
      marker: { size: 6, color },
      x: [point.x, x2],
      y: [point.y, y2],
      hovertemplate: `${label}:<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>`,
    };
  }

  function make2DPointTrace(point, color, label) {
    return {
      type: 'scatter',
      mode: 'markers+text',
      name: label,
      marker: { color, size: 9 },
      text: [label],
      textposition: 'top center',
      x: [point.x],
      y: [point.y],
      hovertemplate: `${label}:<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>`,
    };
  }

  function makeAxesLine(axis, color, name) {
    return {
      type: 'scatter',
      mode: 'lines',
      name,
      line: { color, width: 2, dash: 'dash' },
      x: axis.x,
      y: axis.y,
      hoverinfo: 'skip',
    };
  }

  function makeHyperbolaTrace(value) {
    const samples = 160;
    const xs = [];
    const ys = [];
    const range = 1.5;
    for (let i = 0; i < samples; i += 1) {
      const x = -range + (2 * range * i) / (samples - 1);
      if (Math.abs(x) < 0.2) {
        xs.push(null);
        ys.push(null);
        continue;
      }
      const y = value / x;
      if (Math.abs(y) <= range) {
        xs.push(x);
        ys.push(y);
      } else {
        xs.push(null);
        ys.push(null);
      }
    }
    return {
      type: 'scatter',
      mode: 'lines',
      name: `Level set xy = ${value}`,
      line: { color: '#d62728', width: 3 },
      x: xs,
      y: ys,
      hovertemplate: 'Level set:<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>',
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
    const intersection2dContainer = document.getElementById('lagrange-intersection-2d');
    const tangency2dContainer = document.getElementById('lagrange-tangency-2d');

    if (!intersectionContainer || !tangencyContainer) {
      return;
    }

    const { xVals, yVals, zMatrix } = buildSurfaceData();
    const constraintSamples = circleConstraintSamples(160);
    const circleTrace2D = circle2DTrace(constraintSamples);
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

    if (intersection2dContainer && tangency2dContainer) {
      const axesLines = [
        makeAxesLine({ x: [-1.6, 1.6], y: [0, 0] }, '#ff7f0e', 'Level set f = 0 (y=0)'),
        makeAxesLine({ x: [0, 0], y: [-1.6, 1.6] }, '#ff7f0e', 'Level set f = 0 (x=0)'),
      ];
      const hyperbolaTrace = makeHyperbolaTrace(0.5);

      const base2DLayout = {
        xaxis: { title: 'x', range: [-1.5, 1.5], zeroline: false },
        yaxis: { title: 'y', range: [-1.5, 1.5], zeroline: false, scaleanchor: 'x' },
        margin: { l: 50, r: 20, t: 40, b: 40 },
        legend: { orientation: 'h' },
        height: 420,
      };

      Plotly.newPlot(
        intersection2dContainer,
        [
          circleTrace2D,
          ...axesLines,
          make2DPointTrace(intersectionPoint, '#d62728', 'Intersection point'),
          make2DArrowTrace(intersectionPoint, intersectionPoint.grad, '#d62728', '∇f'),
          make2DArrowTrace(intersectionPoint, intersectionPoint.gradConstraint, '#2ca02c', '∇g'),
        ],
        {
          ...base2DLayout,
          title: 'Intersection in the constraint plane',
        },
        { responsive: true }
      );

      Plotly.newPlot(
        tangency2dContainer,
        [
          circleTrace2D,
          hyperbolaTrace,
          make2DPointTrace(tangencyPoint, '#9467bd', 'Tangency point'),
          make2DArrowTrace(tangencyPoint, tangencyPoint.grad, '#d62728', '∇f'),
          make2DArrowTrace(tangencyPoint, tangencyPoint.gradConstraint, '#2ca02c', '∇g'),
        ],
        {
          ...base2DLayout,
          title: 'Tangency in the constraint plane',
        },
        { responsive: true }
      );
    }

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
