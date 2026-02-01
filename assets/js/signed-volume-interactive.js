// Interactive Signed Volume Visualizations
// Requires Plotly.js for 3D visualizations

document.addEventListener('DOMContentLoaded', function() {
  // Initialize all interactive plots
  if (typeof Plotly !== 'undefined') {
    initOrientation2D();
    initOrientation3D();
    initOrientationTest();
    initSignedAreaCalculator();
  } else {
    console.warn('Plotly.js not loaded. Interactive visualizations will not display.');
  }
});

// 1. 2D Orientation Visualization
function initOrientation2D() {
  const container = document.getElementById('orientation-2d');
  if (!container) return;

  // Initial vectors
  let v1 = { x: 3, y: 0 };
  let v2 = { x: 1, y: 2 };

  function plotOrientation() {
    const det = v1.x * v2.y - v1.y * v2.x;
    const orientation = det > 0 ? 'Counterclockwise (Positive)' : det < 0 ? 'Clockwise (Negative)' : 'Collinear (Zero)';
    const color = det > 0 ? '#2E7D32' : det < 0 ? '#C62828' : '#757575';

    // Create parallelogram
    const parallelogram_x = [0, v1.x, v1.x + v2.x, v2.x, 0];
    const parallelogram_y = [0, v1.y, v1.y + v2.y, v2.y, 0];

    const data = [
      // Parallelogram
      {
        x: parallelogram_x,
        y: parallelogram_y,
        mode: 'lines',
        fill: 'toself',
        fillcolor: color + '40',
        line: { color: color, width: 2 },
        name: 'Parallelogram',
        hoverinfo: 'skip'
      },
      // Vector 1
      {
        x: [0, v1.x],
        y: [0, v1.y],
        mode: 'lines+markers',
        line: { color: '#1976D2', width: 3 },
        marker: { size: 8, symbol: 'arrow', angleref: 'previous' },
        name: 'v₁',
        hovertemplate: 'v₁ = (%{x:.2f}, %{y:.2f})<extra></extra>'
      },
      // Vector 2
      {
        x: [0, v2.x],
        y: [0, v2.y],
        mode: 'lines+markers',
        line: { color: '#D32F2F', width: 3 },
        marker: { size: 8, symbol: 'arrow', angleref: 'previous' },
        name: 'v₂',
        hovertemplate: 'v₂ = (%{x:.2f}, %{y:.2f})<extra></extra>'
      }
    ];

    const layout = {
      title: {
        text: `Signed Area: ${det.toFixed(2)}<br>${orientation}`,
        font: { size: 16 }
      },
      xaxis: { range: [-1, 5], zeroline: true, title: 'x' },
      yaxis: { range: [-1, 4], zeroline: true, title: 'y', scaleanchor: 'x' },
      showlegend: true,
      hovermode: 'closest',
      annotations: [
        {
          x: (v1.x + v2.x) / 2,
          y: (v1.y + v2.y) / 2,
          text: `Area = ${Math.abs(det).toFixed(2)}<br>Sign = ${det > 0 ? '+' : det < 0 ? '−' : '0'}`,
          showarrow: false,
          font: { size: 14, color: color }
        }
      ]
    };

    Plotly.newPlot(container, data, layout, { responsive: true });
  }

  plotOrientation();

  // Add controls
  const controls = document.getElementById('orientation-2d-controls');
  if (controls) {
    controls.innerHTML = `
      <div style="margin: 10px 0;">
        <label>v₁: x = <input type="range" id="v1x" min="-5" max="5" step="0.5" value="3" style="width: 150px;"> 
        <span id="v1x-val">3.0</span></label>
        <label style="margin-left: 20px;">y = <input type="range" id="v1y" min="-5" max="5" step="0.5" value="0" style="width: 150px;"> 
        <span id="v1y-val">0.0</span></label>
      </div>
      <div style="margin: 10px 0;">
        <label>v₂: x = <input type="range" id="v2x" min="-5" max="5" step="0.5" value="1" style="width: 150px;"> 
        <span id="v2x-val">1.0</span></label>
        <label style="margin-left: 20px;">y = <input type="range" id="v2y" min="-5" max="5" step="0.5" value="2" style="width: 150px;"> 
        <span id="v2y-val">2.0</span></label>
      </div>
      <button id="swap-vectors" style="margin: 10px 0; padding: 5px 15px;">Swap Vectors</button>
    `;

    ['v1x', 'v1y', 'v2x', 'v2y'].forEach(id => {
      const input = document.getElementById(id);
      const display = document.getElementById(id + '-val');
      input.addEventListener('input', function() {
        const value = parseFloat(this.value);
        display.textContent = value.toFixed(1);
        if (id.startsWith('v1')) {
          v1[id.charAt(2)] = value;
        } else {
          v2[id.charAt(2)] = value;
        }
        plotOrientation();
      });
    });

    document.getElementById('swap-vectors').addEventListener('click', function() {
      const temp = { ...v1 };
      v1 = { ...v2 };
      v2 = temp;
      document.getElementById('v1x').value = v1.x;
      document.getElementById('v1y').value = v1.y;
      document.getElementById('v2x').value = v2.x;
      document.getElementById('v2y').value = v2.y;
      ['v1x', 'v1y', 'v2x', 'v2y'].forEach(id => {
        const input = document.getElementById(id);
        document.getElementById(id + '-val').textContent = parseFloat(input.value).toFixed(1);
      });
      plotOrientation();
    });
  }
}

// 2. 3D Orientation Visualization
function initOrientation3D() {
  const container = document.getElementById('orientation-3d');
  if (!container) return;

  // Standard basis vectors
  const v1 = [1, 0, 0];
  const v2 = [0, 1, 0];
  const v3 = [0, 0, 1];

  function plot3DOrientation(vectors, title, color) {
    const [v1, v2, v3] = vectors;
    
    // Compute determinant (scalar triple product)
    const det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
                v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
                v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);

    const data = [
      // x-axis (v1)
      {
        type: 'scatter3d',
        mode: 'lines+markers',
        x: [0, v1[0]], y: [0, v1[1]], z: [0, v1[2]],
        line: { color: '#E53935', width: 6 },
        marker: { size: 5, color: '#E53935' },
        name: 'x-axis',
        hovertemplate: 'x = (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
      },
      // y-axis (v2)
      {
        type: 'scatter3d',
        mode: 'lines+markers',
        x: [0, v2[0]], y: [0, v2[1]], z: [0, v2[2]],
        line: { color: '#43A047', width: 6 },
        marker: { size: 5, color: '#43A047' },
        name: 'y-axis',
        hovertemplate: 'y = (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
      },
      // z-axis (v3)
      {
        type: 'scatter3d',
        mode: 'lines+markers',
        x: [0, v3[0]], y: [0, v3[1]], z: [0, v3[2]],
        line: { color: '#1E88E5', width: 6 },
        marker: { size: 5, color: '#1E88E5' },
        name: 'z-axis',
        hovertemplate: 'z = (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
      }
    ];

    const handedness = det > 0 ? 'Right-Handed' : det < 0 ? 'Left-Handed' : 'Degenerate';

    const layout = {
      title: {
        text: `${title}<br>det = ${det.toFixed(2)} (${handedness})`,
        font: { size: 14 }
      },
      scene: {
        xaxis: { range: [-1.5, 1.5], title: 'X' },
        yaxis: { range: [-1.5, 1.5], title: 'Y' },
        zaxis: { range: [-1.5, 1.5], title: 'Z' },
        aspectmode: 'cube',
        camera: {
          eye: { x: 1.5, y: 1.5, z: 1.2 }
        }
      },
      showlegend: true,
      margin: { l: 0, r: 0, t: 40, b: 0 }
    };

    return { data, layout };
  }

  // Right-handed system
  const rightHanded = plot3DOrientation([v1, v2, v3], 'Right-Handed System', '#2E7D32');
  Plotly.newPlot(container, rightHanded.data, rightHanded.layout, { responsive: true });

  // Add toggle button
  const controls = document.getElementById('orientation-3d-controls');
  if (controls) {
    controls.innerHTML = `
      <button id="toggle-handedness" style="margin: 10px 0; padding: 5px 15px;">
        Switch to Left-Handed System
      </button>
    `;

    let isRightHanded = true;
    document.getElementById('toggle-handedness').addEventListener('click', function() {
      isRightHanded = !isRightHanded;
      const vectors = isRightHanded ? 
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]] :  // Right-handed
        [[1, 0, 0], [0, 1, 0], [0, 0, -1]];  // Left-handed (flip z)
      
      const plot = plot3DOrientation(
        vectors,
        isRightHanded ? 'Right-Handed System' : 'Left-Handed System',
        isRightHanded ? '#2E7D32' : '#C62828'
      );
      
      Plotly.react(container, plot.data, plot.layout);
      this.textContent = isRightHanded ? 
        'Switch to Left-Handed System' : 
        'Switch to Right-Handed System';
    });
  }
}

// 3. Interactive Orientation Test
function initOrientationTest() {
  const container = document.getElementById('orientation-test');
  if (!container) return;

  // Points
  let A = { x: 0, y: 0 };
  let B = { x: 4, y: 0 };
  let P = { x: 2, y: 3 };

  function plotTest() {
    // Compute orientation using determinant
    const det = (B.x - A.x) * (P.y - A.y) - (B.y - A.y) * (P.x - A.x);
    const orientation = det > 0 ? 'LEFT (Counterclockwise)' : det < 0 ? 'RIGHT (Clockwise)' : 'ON LINE (Collinear)';
    const color = det > 0 ? '#2E7D32' : det < 0 ? '#C62828' : '#757575';

    const data = [
      // Line AB
      {
        x: [A.x, B.x],
        y: [A.y, B.y],
        mode: 'lines+markers',
        line: { color: '#1976D2', width: 3 },
        marker: { size: 10 },
        name: 'Line AB',
        hovertemplate: 'Point %{text}<extra></extra>',
        text: ['A', 'B']
      },
      // Point P
      {
        x: [P.x],
        y: [P.y],
        mode: 'markers',
        marker: { size: 15, color: color, symbol: 'star' },
        name: 'Point P',
        hovertemplate: 'P = (%{x:.2f}, %{y:.2f})<extra></extra>'
      },
      // Triangle ABP (if not collinear)
      {
        x: [A.x, B.x, P.x, A.x],
        y: [A.y, B.y, P.y, A.y],
        mode: 'lines',
        line: { color: color, width: 1, dash: 'dash' },
        fill: 'toself',
        fillcolor: color + '20',
        name: 'Triangle',
        hoverinfo: 'skip'
      }
    ];

    const layout = {
      title: {
        text: `Point P is ${orientation}<br>Signed Area = ${(det/2).toFixed(2)}`,
        font: { size: 16 }
      },
      xaxis: { range: [-1, 6], zeroline: true, title: 'x' },
      yaxis: { range: [-1, 5], zeroline: true, title: 'y', scaleanchor: 'x' },
      showlegend: true,
      hovermode: 'closest'
    };

    Plotly.newPlot(container, data, layout, { responsive: true });
  }

  plotTest();

  // Add controls
  const controls = document.getElementById('orientation-test-controls');
  if (controls) {
    controls.innerHTML = `
      <div style="margin: 10px 0;">
        <strong>Drag point P to test different positions!</strong>
      </div>
      <div style="margin: 10px 0;">
        <label>P: x = <input type="range" id="px" min="-1" max="6" step="0.5" value="2" style="width: 150px;"> 
        <span id="px-val">2.0</span></label>
        <label style="margin-left: 20px;">y = <input type="range" id="py" min="-1" max="5" step="0.5" value="3" style="width: 150px;"> 
        <span id="py-val">3.0</span></label>
      </div>
    `;

    ['px', 'py'].forEach(id => {
      const input = document.getElementById(id);
      const display = document.getElementById(id + '-val');
      input.addEventListener('input', function() {
        const value = parseFloat(this.value);
        display.textContent = value.toFixed(1);
        P[id.charAt(1)] = value;
        plotTest();
      });
    });
  }
}

// 4. Signed Area Calculator
function initSignedAreaCalculator() {
  const container = document.getElementById('signed-area-calc');
  if (!container) return;

  const form = document.getElementById('area-calc-form');
  const result = document.getElementById('area-calc-result');

  if (!form || !result) return;

  form.addEventListener('submit', function(e) {
    e.preventDefault();

    const v1x = parseFloat(document.getElementById('calc-v1x').value);
    const v1y = parseFloat(document.getElementById('calc-v1y').value);
    const v2x = parseFloat(document.getElementById('calc-v2x').value);
    const v2y = parseFloat(document.getElementById('calc-v2y').value);

    const det = v1x * v2y - v1y * v2x;
    const orientation = det > 0 ? 'Counterclockwise (positive)' : det < 0 ? 'Clockwise (negative)' : 'Collinear (zero)';
    const color = det > 0 ? '#2E7D32' : det < 0 ? '#C62828' : '#757575';

    result.innerHTML = `
      <div style="padding: 15px; background: ${color}20; border-left: 4px solid ${color}; margin-top: 10px;">
        <h4 style="margin-top: 0;">Results:</h4>
        <p><strong>Determinant:</strong> ${det.toFixed(4)}</p>
        <p><strong>Signed Area:</strong> ${det.toFixed(4)} square units</p>
        <p><strong>Physical Area:</strong> ${Math.abs(det).toFixed(4)} square units</p>
        <p><strong>Orientation:</strong> ${orientation}</p>
        <p style="margin-bottom: 0;"><strong>Matrix:</strong><br>
        det[${v1x}, ${v2x}; ${v1y}, ${v2y}] = ${v1x}×${v2y} − ${v2x}×${v1y} = ${det.toFixed(4)}
        </p>
      </div>
    `;
  });

  // Trigger initial calculation
  form.dispatchEvent(new Event('submit'));
}
