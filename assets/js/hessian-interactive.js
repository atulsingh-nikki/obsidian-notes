/* Interactive figures for the Hessian post. */
(function () {
  function linspace(min, max, count) {
    if (count === 1) return [min];
    var step = (max - min) / (count - 1);
    var values = new Array(count);
    for (var i = 0; i < count; i += 1) {
      values[i] = min + step * i;
    }
    return values;
  }

  function buildSurface(zFn) {
    var gridSize = 60;
    var xs = linspace(-2, 2, gridSize);
    var ys = linspace(-2, 2, gridSize);
    var zs = [];
    for (var j = 0; j < ys.length; j += 1) {
      var row = [];
      for (var i = 0; i < xs.length; i += 1) {
        row.push(zFn(xs[i], ys[j]));
      }
      zs.push(row);
    }
    return { xs: xs, ys: ys, zs: zs };
  }

  function ensurePlotly(callback) {
    if (window.Plotly && typeof window.Plotly.newPlot === 'function') {
      callback();
    } else {
      console.warn('Plotly not available; Hessian interactive figures skipped.');
    }
  }

  function renderSurfaces(container) {
    var surfaces = [
      {
        fn: function (x, y) { return x * x + y * y; },
        title: 'Positive definite Hessian',
        colorscale: 'Viridis'
      },
      {
        fn: function (x, y) { return x * x - y * y; },
        title: 'Indefinite Hessian',
        colorscale: 'Plasma'
      },
      {
        fn: function (x, y) { return Math.pow(x, 4) + y * y; },
        title: 'Semidefinite Hessian',
        colorscale: 'Cividis'
      }
    ];

    var traces = [];
    for (var i = 0; i < surfaces.length; i += 1) {
      var surfaceData = buildSurface(surfaces[i].fn);
      traces.push({
        type: 'surface',
        x: surfaceData.xs,
        y: surfaceData.ys,
        z: surfaceData.zs,
        colorscale: surfaces[i].colorscale,
        showscale: false,
        name: surfaces[i].title,
        scene: 'scene' + (i + 1)
      });
    }

    var layout = {
      title: 'Local geometry dictated by the Hessian',
      grid: { rows: 1, columns: 3, pattern: 'independent' },
      margin: { l: 0, r: 0, t: 60, b: 0 },
      height: 500
    };

    for (var idx = 0; idx < surfaces.length; idx += 1) {
      var sceneId = 'scene' + (idx + 1);
      layout[sceneId] = {
        xaxis: { title: 'x', range: [-2, 2] },
        yaxis: { title: 'y', range: [-2, 2] },
        zaxis: { title: 'z' },
        aspectmode: 'cube',
        camera: { eye: { x: 1.3, y: 1.1, z: 0.9 } },
        domain: {
          x: [idx / surfaces.length, (idx + 1) / surfaces.length],
          y: [0, 1]
        },
        title: surfaces[idx].title
      };
    }

    window.Plotly.newPlot(container, traces, layout, { responsive: true });
  }

  function renderLevelset(container) {
    var range = linspace(-1.5, 1.5, 120);
    var zGrid = [];
    for (var j = 0; j < range.length; j += 1) {
      var row = [];
      for (var i = 0; i < range.length; i += 1) {
        var x = range[i];
        var y = range[j];
        row.push(x * x + y * y);
      }
      zGrid.push(row);
    }

    var contour = {
      type: 'contour',
      x: range,
      y: range,
      z: zGrid,
      contours: { showlabels: true, labelfont: { size: 10, color: '#444' } },
      colorscale: 'Greys',
      showscale: false,
      name: 'f(x,y) = x^2 + y^2'
    };

    var theta = linspace(0, Math.PI * 2, 200);
    var circleX = [];
    var circleY = [];
    for (var k = 0; k < theta.length; k += 1) {
      circleX.push(Math.sqrt(0.5) * Math.cos(theta[k]));
      circleY.push(Math.sqrt(0.5) * Math.sin(theta[k]));
    }

    var circleTrace = {
      type: 'scatter',
      mode: 'lines',
      x: circleX,
      y: circleY,
      line: { color: '#1f77b4', width: 3 },
      name: 'Level set f = 0.5'
    };

    var lineRange = linspace(-1.5, 1.5, 80);
    var lineTrace = {
      type: 'scatter',
      mode: 'lines',
      x: lineRange,
      y: lineRange.map(function (v) { return 1 - v; }),
      line: { color: '#ff7f0e', width: 3 },
      name: 'Constraint x + y = 1'
    };

    var pointTrace = {
      type: 'scatter',
      mode: 'markers+text',
      x: [0.5],
      y: [0.5],
      marker: { size: 10, color: '#d62728' },
      text: ['Tangency (0.5, 0.5)'],
      textposition: 'top right',
      showlegend: false
    };

    var layout = {
      title: 'Level-set tangency: f(x,y)=x^2+y^2 with x+y=1',
      xaxis: { title: 'x', scaleanchor: 'y', range: [-1.5, 1.5] },
      yaxis: { title: 'y', range: [-1.5, 1.5] },
      margin: { l: 50, r: 20, t: 60, b: 50 },
      legend: { orientation: 'h' },
      height: 520
    };

    window.Plotly.newPlot(container, [contour, circleTrace, lineTrace, pointTrace], layout, { responsive: true });
  }

  function renderSpectrum(container) {
    var eigenvalues = [5, 2, 1, 0.2, 0.05, 0, 0, 0];
    var xValues = [];
    for (var i = 0; i < eigenvalues.length; i += 1) {
      xValues.push(i + 1);
    }

    var stemTrace = {
      type: 'scatter',
      mode: 'lines+markers',
      x: xValues.reduce(function (acc, val) {
        return acc.concat([val, val, null]);
      }, []),
      y: eigenvalues.reduce(function (acc, val) {
        return acc.concat([0, val, null]);
      }, []),
      line: { color: '#2ca02c', width: 2 },
      marker: { color: '#2ca02c', size: 8 },
      name: 'Eigenvalues'
    };

    var layout = {
      title: 'Example Hessian spectrum (flat directions)',
      xaxis: { title: 'index', dtick: 1 },
      yaxis: { title: 'eigenvalue', rangemode: 'tozero' },
      height: 420,
      margin: { l: 60, r: 20, t: 60, b: 60 },
      showlegend: false
    };

    window.Plotly.newPlot(container, [stemTrace], layout, { responsive: true });
  }

  document.addEventListener('DOMContentLoaded', function () {
    ensurePlotly(function () {
      var surfaceContainer = document.getElementById('hessian-surfaces');
      if (surfaceContainer) {
        renderSurfaces(surfaceContainer);
      }

      var levelsetContainer = document.getElementById('hessian-levelset');
      if (levelsetContainer) {
        renderLevelset(levelsetContainer);
      }

      var spectrumContainer = document.getElementById('hessian-spectrum');
      if (spectrumContainer) {
        renderSpectrum(spectrumContainer);
      }
    });
  });
})();
