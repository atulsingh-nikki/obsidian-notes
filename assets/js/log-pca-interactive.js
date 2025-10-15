(function () {
  function ensurePlotly(cb) {
    if (window.Plotly && typeof window.Plotly.newPlot === 'function') {
      cb();
    } else {
      console.warn('Plotly unavailable; skipping color PCA interactive.');
    }
  }

  function srgbToLinear(channel) {
    var c = channel / 255;
    if (c <= 0.04045) {
      return c / 12.92;
    }
    return Math.pow((c + 0.055) / 1.055, 2.4);
  }

  function linearToSrgb(value) {
    if (value <= 0.0031308) {
      return 255 * 12.92 * value;
    }
    return 255 * (1.055 * Math.pow(value, 1 / 2.4) - 0.055);
  }

  function computeMean(points) {
    var mean = [0, 0, 0];
    for (var i = 0; i < points.length; i += 1) {
      mean[0] += points[i][0];
      mean[1] += points[i][1];
      mean[2] += points[i][2];
    }
    var inv = 1 / points.length;
    mean[0] *= inv;
    mean[1] *= inv;
    mean[2] *= inv;
    return mean;
  }

  function subtractMean(points, mean) {
    var centered = new Array(points.length);
    for (var i = 0; i < points.length; i += 1) {
      centered[i] = [
        points[i][0] - mean[0],
        points[i][1] - mean[1],
        points[i][2] - mean[2]
      ];
    }
    return centered;
  }

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function matVecMul(matrix, vector) {
    return [
      matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
      matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
      matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2]
    ];
  }

  function outer(vecA, vecB) {
    return [
      [vecA[0] * vecB[0], vecA[0] * vecB[1], vecA[0] * vecB[2]],
      [vecA[1] * vecB[0], vecA[1] * vecB[1], vecA[1] * vecB[2]],
      [vecA[2] * vecB[0], vecA[2] * vecB[1], vecA[2] * vecB[2]]
    ];
  }

  function subtractMatrix(a, b) {
    return [
      [a[0][0] - b[0][0], a[0][1] - b[0][1], a[0][2] - b[0][2]],
      [a[1][0] - b[1][0], a[1][1] - b[1][1], a[1][2] - b[1][2]],
      [a[2][0] - b[2][0], a[2][1] - b[2][1], a[2][2] - b[2][2]]
    ];
  }

  function computeCovariance(points) {
    var mean = computeMean(points);
    var centered = subtractMean(points, mean);
    var cov = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ];
    var n = centered.length;
    for (var i = 0; i < n; i += 1) {
      var c = centered[i];
      cov[0][0] += c[0] * c[0];
      cov[0][1] += c[0] * c[1];
      cov[0][2] += c[0] * c[2];
      cov[1][0] += c[1] * c[0];
      cov[1][1] += c[1] * c[1];
      cov[1][2] += c[1] * c[2];
      cov[2][0] += c[2] * c[0];
      cov[2][1] += c[2] * c[1];
      cov[2][2] += c[2] * c[2];
    }
    var inv = 1 / n;
    for (var row = 0; row < 3; row += 1) {
      for (var col = 0; col < 3; col += 1) {
        cov[row][col] *= inv;
      }
    }
    return { mean: mean, covariance: cov };
  }

  function normalize(vec) {
    var length = Math.sqrt(dot(vec, vec));
    if (length === 0) {
      return [0, 0, 0];
    }
    return [vec[0] / length, vec[1] / length, vec[2] / length];
  }

  function powerIteration(matrix, iterations, startVec) {
    var v = startVec || normalize([Math.random(), Math.random(), Math.random()]);
    for (var i = 0; i < iterations; i += 1) {
      var mv = matVecMul(matrix, v);
      var norm = Math.sqrt(dot(mv, mv));
      if (!isFinite(norm) || norm === 0) {
        break;
      }
      v = [mv[0] / norm, mv[1] / norm, mv[2] / norm];
    }
    var eigenvalue = dot(v, matVecMul(matrix, v));
    return { vector: v, value: eigenvalue };
  }

  function computeEigenDecomposition(covariance) {
    var first = powerIteration(covariance, 50);
    var deflated = subtractMatrix(
      covariance,
      outer(first.vector, first.vector).map(function (row, r) {
        return row.map(function (value) { return value * first.value; });
      })
    );

    var second = powerIteration(deflated, 50, normalize([
      first.vector[1],
      -first.vector[0],
      0.0001
    ]));

    // third eigenvector via cross product to maintain orthogonality
    var thirdVec = normalize([
      first.vector[1] * second.vector[2] - first.vector[2] * second.vector[1],
      first.vector[2] * second.vector[0] - first.vector[0] * second.vector[2],
      first.vector[0] * second.vector[1] - first.vector[1] * second.vector[0]
    ]);
    var thirdVal = dot(thirdVec, matVecMul(covariance, thirdVec));

    var eigenpairs = [
      first,
      { vector: second.vector, value: dot(second.vector, matVecMul(covariance, second.vector)) },
      { vector: thirdVec, value: thirdVal }
    ];

    eigenpairs.sort(function (a, b) { return b.value - a.value; });
    return eigenpairs;
  }

  function projectPoints(points, components, mean) {
    var projected = new Array(points.length);
    var centered = subtractMean(points, mean);
    for (var i = 0; i < centered.length; i += 1) {
      var c = centered[i];
      projected[i] = [
        dot(c, components[0]),
        dot(c, components[1]),
        dot(c, components[2])
      ];
    }
    return projected;
  }

  function buildScatter(points, colors, title, axisLabels) {
    var xs = [];
    var ys = [];
    var zs = [];
    var markerColors = colors;
    if (!markerColors) {
      markerColors = points.map(function (p) {
        return 'rgb(' + Math.round(p[0] * 255) + ',' + Math.round(p[1] * 255) + ',' + Math.round(p[2] * 255) + ')';
      });
    }

    for (var i = 0; i < points.length; i += 1) {
      xs.push(points[i][0]);
      ys.push(points[i][1]);
      zs.push(points[i][2]);
    }

    return {
      data: [{
        type: 'scatter3d',
        mode: 'markers',
        x: xs,
        y: ys,
        z: zs,
        marker: {
          size: 3,
          color: markerColors,
          opacity: 0.7
        }
      }],
      layout: {
        title: title,
        margin: { l: 0, r: 0, t: 40, b: 0 },
        height: 360,
        scene: {
          xaxis: { title: axisLabels[0] },
          yaxis: { title: axisLabels[1] },
          zaxis: { title: axisLabels[2] },
          aspectmode: 'cube'
        }
      }
    };
  }

  function multiplyMatrixVector(matrix, vector) {
    return [
      matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
      matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
      matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2]
    ];
  }

  function convertToLogLab(rgbPoints) {
    var toLMS = [
      [0.3811, 0.5783, 0.0402],
      [0.1967, 0.7244, 0.0782],
      [0.0241, 0.1288, 0.8444]
    ];
    var toLab = [
      [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)],
      [1 / Math.sqrt(6), 1 / Math.sqrt(6), -2 / Math.sqrt(6)],
      [1 / Math.sqrt(2), -1 / Math.sqrt(2), 0]
    ];
    var epsilon = 1e-6;
    var transformed = new Array(rgbPoints.length);

    for (var i = 0; i < rgbPoints.length; i += 1) {
      var lms = multiplyMatrixVector(toLMS, rgbPoints[i]);
      var logLms = [
        Math.log10(lms[0] + epsilon),
        Math.log10(lms[1] + epsilon),
        Math.log10(lms[2] + epsilon)
      ];
      transformed[i] = multiplyMatrixVector(toLab, logLms);
    }
    return transformed;
  }

  function renderLogImage(logCanvas, rgbPoints, width, height) {
    var ctx = logCanvas.getContext('2d');
    logCanvas.width = width;
    logCanvas.height = height;
    var imageData = ctx.createImageData(width, height);
    var logScale = Math.log1p(9);
    for (var i = 0; i < rgbPoints.length; i += 1) {
      var linear = rgbPoints[i];
      var logLinear = [
        Math.log1p(9 * linear[0]) / logScale,
        Math.log1p(9 * linear[1]) / logScale,
        Math.log1p(9 * linear[2]) / logScale
      ];
      var r = Math.max(0, Math.min(255, linearToSrgb(logLinear[0])));
      var g = Math.max(0, Math.min(255, linearToSrgb(logLinear[1])));
      var b = Math.max(0, Math.min(255, linearToSrgb(logLinear[2])));
      var idx = i * 4;
      imageData.data[idx] = r;
      imageData.data[idx + 1] = g;
      imageData.data[idx + 2] = b;
      imageData.data[idx + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  }

  function renderOriginalImageData(canvas, imageData, width, height) {
    var ctx = canvas.getContext('2d');
    canvas.width = width;
    canvas.height = height;
    ctx.putImageData(imageData, 0, 0);
  }

  function collectLinearRgb(ctx, width, height) {
    var data = ctx.getImageData(0, 0, width, height).data;
    var points = new Array(width * height);
    for (var i = 0; i < points.length; i += 1) {
      var idx = i * 4;
      points[i] = [
        srgbToLinear(data[idx]),
        srgbToLinear(data[idx + 1]),
        srgbToLinear(data[idx + 2])
      ];
    }
    return points;
  }

  function paintSamplePattern(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);

    var baseGradient = ctx.createLinearGradient(0, 0, width, height);
    baseGradient.addColorStop(0, 'rgb(35, 90, 240)');
    baseGradient.addColorStop(0.45, 'rgb(245, 210, 60)');
    baseGradient.addColorStop(1, 'rgb(210, 55, 140)');
    ctx.fillStyle = baseGradient;
    ctx.fillRect(0, 0, width, height);

    var radialWarm = ctx.createRadialGradient(width * 0.25, height * 0.3, 6, width * 0.25, height * 0.3, width * 0.55);
    radialWarm.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
    radialWarm.addColorStop(0.6, 'rgba(255, 170, 90, 0.55)');
    radialWarm.addColorStop(1, 'rgba(255, 120, 70, 0)');
    ctx.fillStyle = radialWarm;
    ctx.fillRect(0, 0, width, height);

    var radialCool = ctx.createRadialGradient(width * 0.72, height * 0.72, 8, width * 0.72, height * 0.72, width * 0.45);
    radialCool.addColorStop(0, 'rgba(60, 220, 255, 0.9)');
    radialCool.addColorStop(0.7, 'rgba(40, 120, 210, 0.35)');
    radialCool.addColorStop(1, 'rgba(40, 120, 210, 0)');
    ctx.fillStyle = radialCool;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.globalAlpha = 0.35;
    for (var x = 0; x < width; x += 12) {
      ctx.fillStyle = x % 24 === 0 ? 'rgba(255, 87, 34, 0.9)' : 'rgba(0, 139, 139, 0.85)';
      ctx.fillRect(x, 0, 6, height);
    }
    ctx.restore();

    ctx.save();
    ctx.translate(width / 2, height / 2);
    ctx.rotate(-Math.PI / 6);
    ctx.globalAlpha = 0.6;
    var ribbon = ctx.createLinearGradient(-width, 0, width, 0);
    ribbon.addColorStop(0, 'rgba(255, 0, 170, 0.95)');
    ribbon.addColorStop(0.5, 'rgba(255, 255, 255, 0.4)');
    ribbon.addColorStop(1, 'rgba(80, 200, 255, 0.9)');
    ctx.fillStyle = ribbon;
    ctx.fillRect(-width, -height * 0.15, width * 2, height * 0.3);
    ctx.restore();

    ctx.save();
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = 'rgba(40, 40, 40, 0.85)';
    ctx.fillRect(width * 0.05, height * 0.65, width * 0.18, height * 0.25);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.fillRect(width * 0.75, height * 0.1, width * 0.2, height * 0.2);
    ctx.restore();

    ctx.beginPath();
    ctx.fillStyle = 'rgba(30, 200, 255, 0.85)';
    ctx.arc(width * 0.6, height * 0.35, height * 0.18, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.fillStyle = 'rgba(255, 220, 35, 0.85)';
    ctx.moveTo(width * 0.35, height * 0.15);
    ctx.lineTo(width * 0.48, height * 0.42);
    ctx.lineTo(width * 0.18, height * 0.42);
    ctx.closePath();
    ctx.fill();
  }

  document.addEventListener('DOMContentLoaded', function () {
    var container = document.getElementById('lab-color-playground');
    if (!container) {
      return;
    }

    var originalCanvas = document.getElementById('color-pca-original');
    var logCanvas = document.getElementById('color-pca-log');
    var rgbPlot = document.getElementById('color-pca-rgb');
    var pcaPlot = document.getElementById('color-pca-pca');
    var logPlot = document.getElementById('color-pca-logplot');
    var statsBox = document.getElementById('color-pca-stats');

    if (!originalCanvas || !logCanvas || !rgbPlot || !pcaPlot || !logPlot) {
      console.warn('Missing elements for color PCA demo.');
      return;
    }

    var sampleWidth = 96;
    var sampleHeight = 96;
    var scratch = document.createElement('canvas');
    scratch.width = sampleWidth;
    scratch.height = sampleHeight;
    var scratchCtx = scratch.getContext('2d');
    paintSamplePattern(scratchCtx, sampleWidth, sampleHeight);

    var imageData = scratchCtx.getImageData(0, 0, sampleWidth, sampleHeight);
    renderOriginalImageData(originalCanvas, imageData, sampleWidth, sampleHeight);

    var linearPoints = collectLinearRgb(scratchCtx, sampleWidth, sampleHeight);
    var srgbColors = linearPoints.map(function (p) {
      var r = Math.max(0, Math.min(255, linearToSrgb(p[0])));
      var g = Math.max(0, Math.min(255, linearToSrgb(p[1])));
      var b = Math.max(0, Math.min(255, linearToSrgb(p[2])));
      return 'rgb(' + Math.round(r) + ',' + Math.round(g) + ',' + Math.round(b) + ')';
    });
    renderLogImage(logCanvas, linearPoints, sampleWidth, sampleHeight);

    var covInfo = computeCovariance(linearPoints);
    var eigenpairs = computeEigenDecomposition(covInfo.covariance);
    var components = [eigenpairs[0].vector, eigenpairs[1].vector, eigenpairs[2].vector];
    var projected = projectPoints(linearPoints, components, covInfo.mean);
    var logLabPoints = convertToLogLab(linearPoints);

    ensurePlotly(function () {
      var rgbScatter = buildScatter(linearPoints, srgbColors, 'RGB distribution (linear light)', ['R', 'G', 'B']);
      window.Plotly.newPlot(rgbPlot, rgbScatter.data, rgbScatter.layout, { responsive: true });

      var pcaScatter = buildScatter(projected, srgbColors, 'Rotated PCA coordinates', ['PC1', 'PC2', 'PC3']);
      window.Plotly.newPlot(pcaPlot, pcaScatter.data, pcaScatter.layout, { responsive: true });

      var labScatter = buildScatter(logLabPoints, srgbColors, 'Log lαβ approximation', ['l', 'α', 'β']);
      window.Plotly.newPlot(logPlot, labScatter.data, labScatter.layout, { responsive: true });
    });

    if (statsBox) {
      var varianceSum = eigenpairs[0].value + eigenpairs[1].value + eigenpairs[2].value;
      var lines = [
        '<strong>Mean (linear RGB):</strong> ' + covInfo.mean.map(function (v) { return v.toFixed(3); }).join(', '),
        '<strong>Eigenvalues:</strong> ' + eigenpairs.map(function (p) { return p.value.toFixed(5); }).join(' • '),
        '<strong>Variance captured by PC1:</strong> ' + ((eigenpairs[0].value / varianceSum) * 100).toFixed(1) + '%'
      ];
      statsBox.innerHTML = lines.join('<br>');
    }
  });
})();
