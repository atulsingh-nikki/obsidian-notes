/**
 * Interactive Depth vs Disparity Visualization
 * Demonstrates the relationship: Z = (b * f) / d
 * where Z = depth, b = baseline, f = focal length, d = disparity
 */

class DepthDisparityVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container ${containerId} not found`);
            return;
        }
        
        // Default parameters
        this.baseline = 0.12; // meters (typical stereo camera baseline)
        this.focalLength = 700; // pixels
        this.currentDisparity = 50; // pixels
        this.maxDisparity = 200;
        this.minDisparity = 1;
        
        this.init();
        this.createControls();
        this.createVisualization();
        this.update();
    }
    
    init() {
        this.container.innerHTML = `
            <div class="depth-disparity-container">
                <div class="controls-section">
                    <h3>Camera Parameters</h3>
                    <div class="control-group">
                        <label for="baseline-slider">
                            Baseline (b): <span id="baseline-value">0.12</span> m
                        </label>
                        <input type="range" id="baseline-slider" min="0.05" max="1.0" step="0.01" value="0.12">
                        <small>Distance between stereo cameras</small>
                    </div>
                    
                    <div class="control-group">
                        <label for="focal-slider">
                            Focal Length (f): <span id="focal-value">700</span> pixels
                        </label>
                        <input type="range" id="focal-slider" min="300" max="1500" step="10" value="700">
                        <small>Camera focal length in pixels</small>
                    </div>
                    
                    <div class="control-group">
                        <label for="disparity-slider">
                            Disparity (d): <span id="disparity-value">50</span> pixels
                        </label>
                        <input type="range" id="disparity-slider" min="1" max="200" step="1" value="50">
                        <small>Pixel offset between stereo images</small>
                    </div>
                    
                    <div class="result-box">
                        <h4>Computed Depth</h4>
                        <div class="depth-result">
                            Z = <span id="depth-value">1.68</span> m
                        </div>
                        <div class="formula">
                            Z = (b × f) / d
                        </div>
                    </div>
                </div>
                
                <div class="visualization-section">
                    <div class="canvas-container">
                        <h3>3D Stereo Camera Geometry</h3>
                        <canvas id="3d-canvas" width="600" height="350"></canvas>
                    </div>
                    
                    <div class="canvas-container">
                        <h3>Stereo Camera Setup (Top View)</h3>
                        <canvas id="stereo-canvas" width="600" height="300"></canvas>
                    </div>
                    
                    <div class="graph-container">
                        <h3>Depth vs Disparity Relationship</h3>
                        <canvas id="graph-canvas" width="600" height="300"></canvas>
                    </div>
                </div>
                
                <div class="insights-section">
                    <h4>Key Insights:</h4>
                    <ul id="insights-list">
                        <li>Depth is inversely proportional to disparity: as disparity increases, depth decreases</li>
                        <li>Larger baseline → larger disparity → better depth resolution for far objects</li>
                        <li>Depth uncertainty grows quadratically: σ_Z ∝ Z²</li>
                    </ul>
                </div>
            </div>
        `;
        
        this.addStyles();
    }
    
    addStyles() {
        if (document.getElementById('depth-disparity-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'depth-disparity-styles';
        style.textContent = `
            .depth-disparity-container {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 1400px;
                margin: 2rem auto;
                padding: 2rem;
                background: #f8f9fa;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                display: grid;
                grid-template-columns: 350px 1fr;
                gap: 2rem;
            }
            
            .controls-section {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
            }
            
            .controls-section h3 {
                margin-top: 0;
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 0.5rem;
            }
            
            .control-group {
                margin: 1.5rem 0;
            }
            
            .control-group label {
                display: block;
                font-weight: 600;
                color: #34495e;
                margin-bottom: 0.5rem;
            }
            
            .control-group input[type="range"] {
                width: 100%;
                height: 8px;
                border-radius: 5px;
                background: #dfe6e9;
                outline: none;
                -webkit-appearance: none;
            }
            
            .control-group input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #3498db;
                cursor: pointer;
                transition: background 0.3s;
            }
            
            .control-group input[type="range"]::-webkit-slider-thumb:hover {
                background: #2980b9;
            }
            
            .control-group input[type="range"]::-moz-range-thumb {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #3498db;
                cursor: pointer;
                border: none;
            }
            
            .control-group small {
                display: block;
                color: #7f8c8d;
                font-size: 0.85rem;
                margin-top: 0.25rem;
            }
            
            .result-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 8px;
                color: white;
                margin-top: 2rem;
            }
            
            .result-box h4 {
                margin: 0 0 1rem 0;
                font-size: 1.1rem;
            }
            
            .depth-result {
                font-size: 2rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }
            
            .formula {
                font-family: 'Courier New', monospace;
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .visualization-section {
                display: grid;
                grid-template-columns: 1fr;
                gap: 2rem;
            }
            
            .canvas-container, .graph-container {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
            }
            
            .canvas-container h3, .graph-container h3 {
                margin-top: 0;
                color: #2c3e50;
                border-bottom: 2px solid #e74c3c;
                padding-bottom: 0.5rem;
            }
            
            canvas {
                display: block;
                width: 100%;
                border: 1px solid #ecf0f1;
                border-radius: 4px;
            }
            
            .insights-section {
                background: #fff3cd;
                padding: 1.5rem;
                border-radius: 8px;
                border-left: 4px solid #ffc107;
                margin-top: 2rem;
                grid-column: 1 / -1;
            }
            
            .insights-section h4 {
                margin-top: 0;
                color: #856404;
            }
            
            .insights-section ul {
                margin: 0;
                padding-left: 1.5rem;
                color: #856404;
            }
            
            .insights-section li {
                margin: 0.5rem 0;
            }
            
            @media (max-width: 1024px) {
                .depth-disparity-container {
                    grid-template-columns: 1fr;
                    gap: 1.5rem;
                }
                
                .insights-section {
                    grid-column: 1;
                }
            }
            
            @media (max-width: 768px) {
                .depth-disparity-container {
                    padding: 1rem;
                }
                
                .depth-result {
                    font-size: 1.5rem;
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    createControls() {
        const baselineSlider = document.getElementById('baseline-slider');
        const focalSlider = document.getElementById('focal-slider');
        const disparitySlider = document.getElementById('disparity-slider');
        
        baselineSlider.addEventListener('input', (e) => {
            this.baseline = parseFloat(e.target.value);
            document.getElementById('baseline-value').textContent = this.baseline.toFixed(2);
            this.update();
        });
        
        focalSlider.addEventListener('input', (e) => {
            this.focalLength = parseFloat(e.target.value);
            document.getElementById('focal-value').textContent = this.focalLength.toFixed(0);
            this.update();
        });
        
        disparitySlider.addEventListener('input', (e) => {
            this.currentDisparity = parseFloat(e.target.value);
            document.getElementById('disparity-value').textContent = this.currentDisparity.toFixed(0);
            this.update();
        });
    }
    
    createVisualization() {
        this.threeDCanvas = document.getElementById('3d-canvas');
        this.stereoCanvas = document.getElementById('stereo-canvas');
        this.graphCanvas = document.getElementById('graph-canvas');
        this.threeDCtx = this.threeDCanvas.getContext('2d');
        this.stereoCtx = this.stereoCanvas.getContext('2d');
        this.graphCtx = this.graphCanvas.getContext('2d');
    }
    
    calculateDepth() {
        return (this.baseline * this.focalLength) / this.currentDisparity;
    }
    
    update() {
        const depth = this.calculateDepth();
        document.getElementById('depth-value').textContent = depth.toFixed(2);
        
        this.draw3DGeometry(depth);
        this.drawStereoSetup(depth);
        this.drawDepthDisparityGraph();
        this.updateInsights(depth);
    }
    
    draw3DGeometry(depth) {
        const ctx = this.threeDCtx;
        const width = this.threeDCanvas.width;
        const height = this.threeDCanvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // 3D to 2D projection parameters (isometric-ish view)
        const scale = 50;
        const originX = width / 2;
        const originY = height * 0.7;
        
        // Projection angles
        const angleX = Math.PI / 6; // 30 degrees
        const angleY = Math.PI / 4; // 45 degrees
        
        // Project 3D point to 2D
        const project = (x, y, z) => {
            const px = originX + (x * Math.cos(angleY) - z * Math.sin(angleY)) * scale;
            const py = originY - (y * Math.cos(angleX) + (x * Math.sin(angleY) + z * Math.cos(angleY)) * Math.sin(angleX)) * scale;
            return { x: px, y: py };
        };
        
        // Camera and point positions in 3D space
        const baselineHalf = this.baseline / 2;
        const leftCam = { x: -baselineHalf, y: 0, z: 0 };
        const rightCam = { x: baselineHalf, y: 0, z: 0 };
        const point3D = { x: 0, y: 0, z: depth };
        
        // Draw coordinate system
        ctx.strokeStyle = '#ddd';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        
        // Z-axis (depth)
        let p1 = project(0, 0, 0);
        let p2 = project(0, 0, Math.min(depth + 1, 6));
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        
        // X-axis (baseline direction)
        p1 = project(-this.baseline, 0, 0);
        p2 = project(this.baseline, 0, 0);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        
        ctx.setLineDash([]);
        
        // Draw baseline connection
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 3;
        p1 = project(leftCam.x, leftCam.y, leftCam.z);
        p2 = project(rightCam.x, rightCam.y, rightCam.z);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        
        // Draw cameras as 3D cones
        this.draw3DCamera(ctx, project, leftCam, point3D, 'Left', depth);
        this.draw3DCamera(ctx, project, rightCam, point3D, 'Right', depth);
        
        // Draw 3D point
        const pPoint = project(point3D.x, point3D.y, point3D.z);
        ctx.fillStyle = '#e74c3c';
        ctx.beginPath();
        ctx.arc(pPoint.x, pPoint.y, 10, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw sight lines with disparity visualization
        ctx.strokeStyle = 'rgba(52, 152, 219, 0.6)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        
        p1 = project(leftCam.x, leftCam.y, leftCam.z);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(pPoint.x, pPoint.y);
        ctx.stroke();
        
        p1 = project(rightCam.x, rightCam.y, rightCam.z);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(pPoint.x, pPoint.y);
        ctx.stroke();
        
        ctx.setLineDash([]);
        
        // Draw image planes (simplified)
        this.draw3DImagePlane(ctx, project, leftCam, depth, 'left');
        this.draw3DImagePlane(ctx, project, rightCam, depth, 'right');
        
        // Add labels
        ctx.font = 'bold 14px Arial';
        ctx.fillStyle = '#2c3e50';
        ctx.textAlign = 'center';
        ctx.fillText('3D Point', pPoint.x, pPoint.y - 15);
        
        ctx.font = '12px Arial';
        ctx.fillStyle = '#7f8c8d';
        const depthLabelPos = project(0, 0, depth / 2);
        ctx.fillText(`Z = ${depth.toFixed(2)}m`, depthLabelPos.x + 30, depthLabelPos.y);
        
        const baselineLabelPos = project(0, -0.15, 0);
        ctx.fillText(`Baseline = ${this.baseline.toFixed(2)}m`, baselineLabelPos.x, baselineLabelPos.y);
    }
    
    draw3DCamera(ctx, project, camPos, point3D, label, depth) {
        const p = project(camPos.x, camPos.y, camPos.z);
        
        // Draw camera body
        ctx.fillStyle = '#34495e';
        ctx.fillRect(p.x - 10, p.y - 8, 20, 16);
        
        // Draw lens
        ctx.fillStyle = '#2c3e50';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw focal length indicator
        const focalLengthWorld = this.focalLength / 1000; // Convert pixels to rough world units
        const focalPoint = project(camPos.x, camPos.y, camPos.z + focalLengthWorld);
        
        ctx.strokeStyle = '#9b59b6';
        ctx.lineWidth = 2;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(focalPoint.x, focalPoint.y);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Label
        ctx.font = 'bold 12px Arial';
        ctx.fillStyle = '#34495e';
        ctx.textAlign = 'center';
        ctx.fillText(label, p.x, p.y + 25);
        
        ctx.font = '10px Arial';
        ctx.fillStyle = '#9b59b6';
        ctx.fillText('f', (p.x + focalPoint.x) / 2 + 10, (p.y + focalPoint.y) / 2);
    }
    
    draw3DImagePlane(ctx, project, camPos, depth, side) {
        // Draw a small rectangle representing the image plane
        const planeHeight = 0.3;
        const planeZ = camPos.z + 0.1;
        
        const corners = [
            project(camPos.x - 0.15, camPos.y - planeHeight/2, planeZ),
            project(camPos.x + 0.15, camPos.y - planeHeight/2, planeZ),
            project(camPos.x + 0.15, camPos.y + planeHeight/2, planeZ),
            project(camPos.x - 0.15, camPos.y + planeHeight/2, planeZ)
        ];
        
        ctx.strokeStyle = 'rgba(149, 165, 166, 0.5)';
        ctx.fillStyle = 'rgba(236, 240, 241, 0.3)';
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        ctx.moveTo(corners[0].x, corners[0].y);
        for (let i = 1; i < corners.length; i++) {
            ctx.lineTo(corners[i].x, corners[i].y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    }
    
    drawStereoSetup(depth) {
        const ctx = this.stereoCtx;
        const width = this.stereoCanvas.width;
        const height = this.stereoCanvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Scale factor for visualization
        const scale = 60; // pixels per meter
        const baselinePixels = this.baseline * scale;
        
        // Camera positions (top view)
        const leftCamX = width / 2 - baselinePixels / 2;
        const rightCamX = width / 2 + baselinePixels / 2;
        const camY = height - 50;
        
        // Scale depth for visualization (cap at canvas width)
        const maxDepth = 8; // meters
        const depthPixels = Math.min(depth * scale, height - 100);
        const pointX = width / 2;
        const pointY = camY - depthPixels;
        
        // Draw baseline
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(leftCamX, camY);
        ctx.lineTo(rightCamX, camY);
        ctx.stroke();
        
        // Draw cameras
        this.drawCamera(ctx, leftCamX, camY, 'Left');
        this.drawCamera(ctx, rightCamX, camY, 'Right');
        
        // Draw 3D point
        ctx.fillStyle = '#e74c3c';
        ctx.beginPath();
        ctx.arc(pointX, pointY, 8, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw sight lines
        ctx.strokeStyle = '#95a5a6';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        
        ctx.beginPath();
        ctx.moveTo(leftCamX, camY);
        ctx.lineTo(pointX, pointY);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(rightCamX, camY);
        ctx.lineTo(pointX, pointY);
        ctx.stroke();
        
        ctx.setLineDash([]);
        
        // Draw depth line
        ctx.strokeStyle = '#27ae60';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(pointX, camY);
        ctx.lineTo(pointX, pointY);
        ctx.stroke();
        
        // Labels
        ctx.fillStyle = '#2c3e50';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        
        // Baseline label
        ctx.fillText(`Baseline: ${this.baseline.toFixed(2)}m`, width / 2, camY + 30);
        
        // Depth label
        ctx.save();
        ctx.translate(pointX + 25, (camY + pointY) / 2);
        ctx.fillStyle = '#27ae60';
        ctx.fillText(`Z = ${depth.toFixed(2)}m`, 0, 0);
        ctx.restore();
        
        // Point label
        ctx.fillStyle = '#e74c3c';
        ctx.fillText('3D Point', pointX, pointY - 15);
        
        // Draw disparity visualization on image plane
        const imagePlaneY = camY - 30;
        const disparityScale = 0.5; // Visual scaling
        const leftImageX = leftCamX - this.currentDisparity * disparityScale;
        const rightImageX = rightCamX + this.currentDisparity * disparityScale;
        
        // Image planes (small lines representing where point projects)
        ctx.strokeStyle = '#9b59b6';
        ctx.lineWidth = 3;
        
        ctx.beginPath();
        ctx.moveTo(leftImageX, imagePlaneY - 10);
        ctx.lineTo(leftImageX, imagePlaneY + 10);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(rightImageX, imagePlaneY - 10);
        ctx.lineTo(rightImageX, imagePlaneY + 10);
        ctx.stroke();
        
        // Disparity arrow
        ctx.strokeStyle = '#9b59b6';
        ctx.lineWidth = 2;
        this.drawArrow(ctx, rightImageX, imagePlaneY - 20, leftImageX, imagePlaneY - 20);
        
        ctx.fillStyle = '#9b59b6';
        ctx.font = '12px Arial';
        ctx.fillText(`d = ${this.currentDisparity.toFixed(0)}px`, (leftImageX + rightImageX) / 2, imagePlaneY - 25);
    }
    
    drawCamera(ctx, x, y, label) {
        // Draw camera body
        ctx.fillStyle = '#34495e';
        ctx.fillRect(x - 15, y - 10, 30, 20);
        
        // Draw lens
        ctx.fillStyle = '#2c3e50';
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fill();
        
        // Label
        ctx.fillStyle = '#2c3e50';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(label, x, y + 30);
    }
    
    drawArrow(ctx, fromX, fromY, toX, toY) {
        const headLength = 10;
        const angle = Math.atan2(toY - fromY, toX - fromX);
        
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), 
                   toY - headLength * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), 
                   toY - headLength * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fillStyle = ctx.strokeStyle;
        ctx.fill();
    }
    
    drawDepthDisparityGraph() {
        const ctx = this.graphCtx;
        const width = this.graphCanvas.width;
        const height = this.graphCanvas.height;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Margins
        const margin = { top: 20, right: 30, bottom: 50, left: 60 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = height - margin.top - margin.bottom;
        
        // Axes
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 2;
        
        // X-axis
        ctx.beginPath();
        ctx.moveTo(margin.left, height - margin.bottom);
        ctx.lineTo(width - margin.right, height - margin.bottom);
        ctx.stroke();
        
        // Y-axis
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top);
        ctx.lineTo(margin.left, height - margin.bottom);
        ctx.stroke();
        
        // Labels
        ctx.fillStyle = '#2c3e50';
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Disparity (pixels)', width / 2, height - 10);
        
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Depth (meters)', 0, 0);
        ctx.restore();
        
        // Draw depth curve: Z = (b*f)/d
        const maxDepth = 10; // meters
        const maxDisp = this.maxDisparity;
        
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 3;
        ctx.beginPath();
        
        for (let d = 1; d <= maxDisp; d += 0.5) {
            const z = (this.baseline * this.focalLength) / d;
            if (z > maxDepth) continue;
            
            const x = margin.left + (d / maxDisp) * plotWidth;
            const y = height - margin.bottom - (z / maxDepth) * plotHeight;
            
            if (d === 1) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();
        
        // Draw current point
        const currentX = margin.left + (this.currentDisparity / maxDisp) * plotWidth;
        const currentDepth = this.calculateDepth();
        const currentY = height - margin.bottom - Math.min(currentDepth, maxDepth) / maxDepth * plotHeight;
        
        ctx.fillStyle = '#e74c3c';
        ctx.beginPath();
        ctx.arc(currentX, currentY, 8, 0, Math.PI * 2);
        ctx.fill();
        
        // Crosshair
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        
        ctx.beginPath();
        ctx.moveTo(currentX, height - margin.bottom);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(margin.left, currentY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
        
        ctx.setLineDash([]);
        
        // Draw tick marks and labels
        ctx.fillStyle = '#2c3e50';
        ctx.font = '11px Arial';
        ctx.textAlign = 'center';
        
        // X-axis ticks
        for (let i = 0; i <= 5; i++) {
            const d = (maxDisp / 5) * i;
            const x = margin.left + (d / maxDisp) * plotWidth;
            
            ctx.beginPath();
            ctx.moveTo(x, height - margin.bottom);
            ctx.lineTo(x, height - margin.bottom + 5);
            ctx.stroke();
            
            ctx.fillText(d.toFixed(0), x, height - margin.bottom + 20);
        }
        
        // Y-axis ticks
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const z = (maxDepth / 5) * i;
            const y = height - margin.bottom - (z / maxDepth) * plotHeight;
            
            ctx.beginPath();
            ctx.moveTo(margin.left - 5, y);
            ctx.lineTo(margin.left, y);
            ctx.stroke();
            
            ctx.fillText(z.toFixed(1), margin.left - 10, y + 4);
        }
        
        // Add formula on graph
        ctx.fillStyle = '#3498db';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Z = (b × f) / d', margin.left + 20, margin.top + 20);
    }
    
    updateInsights(depth) {
        const insights = [];
        
        if (this.currentDisparity < 10) {
            insights.push('⚠️ Very small disparity: Depth estimation becomes unreliable and sensitive to noise');
        } else if (this.currentDisparity > 150) {
            insights.push('✓ Large disparity: Object is close, depth estimate is more accurate');
        }
        
        if (this.baseline < 0.1) {
            insights.push('⚠️ Small baseline: Limited depth range, better for close objects');
        } else if (this.baseline > 0.5) {
            insights.push('✓ Large baseline: Better depth resolution for distant objects');
        }
        
        if (depth < 1) {
            insights.push('📍 Object is very close (< 1m): Suitable for tabletop robotics');
        } else if (depth > 5) {
            insights.push('📍 Object is far (> 5m): Depth uncertainty increases quadratically');
        }
        
        const depthUncertainty = (depth * depth) / (this.baseline * this.focalLength);
        insights.push(`📊 Depth uncertainty factor: ${depthUncertainty.toFixed(2)} (higher = less accurate)`);
        
        const insightsList = document.getElementById('insights-list');
        if (insights.length > 0) {
            let html = '<li>Depth is inversely proportional to disparity: as disparity increases, depth decreases</li>';
            insights.forEach(insight => {
                html += `<li>${insight}</li>`;
            });
            insightsList.innerHTML = html;
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new DepthDisparityVisualizer('depth-disparity-viz');
    });
} else {
    new DepthDisparityVisualizer('depth-disparity-viz');
}
