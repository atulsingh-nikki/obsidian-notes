### Chapter 7: Color Correction for Still Photography
The professional photography workflow is built on the principles of non-destructive editing and rigorous color management to ensure the highest possible quality and fidelity to the photographer's creative vision.

#### 7.1 The Professional RAW Workflow
Working with RAW files is non-negotiable for professional results.42 The workflow, typically executed in software like Adobe Camera Raw (ACR) or Lightroom, is a phased approach designed for efficiency and precision.40

*   **Phase 1: Culling and Global Adjustments**: The process begins with organization and selection. A large shoot is first culled to select the best images. Then, a representative "hero" image from a set taken under consistent lighting is chosen. Initial global corrections are applied to this image:
    *   **White Balance**: Set a neutral starting point using the eyedropper on a gray area or by adjusting temperature/tint sliders.
    *   **Exposure and Contrast**: Use the histogram to set the black and white points, ensuring a full tonal range without clipping.
    *   **Highlight and Shadow Recovery**: Use dedicated sliders to bring back detail in the brightest and darkest areas of the image.
    Once the hero image is corrected, these settings are synchronized or copied to all other images from the same lighting setup, performing the bulk of the basic correction in a single step.40
*   **Phase 2: Local Adjustments and Fine-Tuning**: With the global baseline established, each image is then refined individually. This involves using local adjustment tools like brushes and gradients to perform targeted enhancements, such as brightening a subject's face, darkening a sky, or adding clarity to specific textures. Advanced steps like lens correction, sharpening, and noise reduction are also applied during this phase, often using masks to target their effects precisely.42 This non-destructive process ensures that the original RAW data is always preserved, and any adjustment can be revisited and modified at any time.41

#### 7.2 The Color-Managed Ecosystem
Color management is the overarching system that ensures color appears predictable and consistent across every device in the workflow, from camera to monitor to printer.62 It is a system of translation, not a guarantee of a perfect 1:1 match. A monitor uses transmitted light while a print uses reflected light, leading to inherent differences in dynamic range and appearance.65 The goal of color management is therefore to establish a predictable and controlled system of translation that preserves the intent of the colors across an imperfect system. This system is built on three pillars:

*   **Calibration**: This is the process of adjusting a device to a known, standardized state. For a professional photographer, monitor calibration is the absolute cornerstone of the entire workflow. It is performed using a hardware device—a colorimeter or spectrophotometer—that measures the color output of the screen and works with software to adjust the monitor's settings and graphics card to match industry standards (e.g., a specific white point like D65 and gamma).20 An uncalibrated monitor provides a false representation of the image, making accurate editing impossible.
*   **Profiling**: After calibration, a device is profiled. This process involves measuring how the device reproduces a standard set of colors. The software then creates a unique ICC (International Color Consortium) profile, which is a data file that describes the device's specific color gamut and characteristics.20 This profile acts as a "translator" or "dictionary" for that device's unique color language. A complete workflow requires profiles for the camera, monitor, and each specific printer/paper combination.
*   **Conversion**: The color management module (CMM) of the operating system or application uses these ICC profiles to translate colors between devices. When an image is opened, the CMM reads its embedded profile (e.g., Adobe RGB) and the monitor's profile, then converts the image data on-the-fly to display it accurately on that specific screen. This ensures that the colors seen on a calibrated and profiled monitor are a reliable representation of the actual data in the file.64

#### 7.3 Output for Print and Web
The final stage of the workflow is preparing the image for its intended destination, which requires specific color space conversions and proofing techniques.

*   **Soft Proofing**: This is a critical step for print preparation. Using the specific ICC profile for the target printer and paper, the editing software can simulate on-screen how the final image will look when printed.20 This on-screen preview reveals how the image's colors will shift when converted to the smaller CMYK gamut of the printer, allowing the photographer to make targeted adjustments (e.g., reducing saturation in out-of-gamut colors) to optimize the file for printing before any ink or paper is wasted.68
*   **Exporting with the Correct Profile**: The choice of color space for the final exported file is crucial.
    *   **For Web and Mobile**: Images should be converted to and saved in the sRGB color space. Since sRGB is the standard for the internet, this ensures the colors will appear as intended for the widest possible audience on unmanaged displays.25
    *   **For High-Quality Printing**: When sending files to a high-end lab or printing on a professional inkjet printer, embedding a wider-gamut profile like Adobe RGB (1998) is often preferred, as these devices can reproduce a larger range of colors than sRGB allows.25 The file is then converted to the printer's specific CMYK profile by the print service provider's RIP (Raster Image Processor) software.

### References
20. [Citation text to be added]
25. [Citation text to be added]
40. [Citation text to be added]
41. [Citation text to be added]
42. [Citation text to be added]
62. [Citation text to be added]
64. [Citation text to be added]
65. [Citation text to be added]
68. [Citation text to be added]
