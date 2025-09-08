### Chapter 10: High Dynamic Range (HDR) and Tone Mapping
The advent of High Dynamic Range (HDR) technology represents one of the most significant advancements in digital imaging since the transition to color. It allows for the capture and display of a much wider range of luminance and color, resulting in images that are more vibrant, realistic, and closer to the perception of the human eye.

#### 10.1 Fundamentals of High Dynamic Range Imaging
Dynamic Range refers to the ratio between the brightest and darkest parts of an image a device can capture or display.86 Standard Dynamic Range (SDR) imaging, based on legacy CRT display technology, is limited in the range of light it can represent. HDR technology dramatically expands this range, allowing for specular highlights that are significantly brighter and shadows that are deeper and more detailed, all within the same image.39

*   **HDR Acquisition**: Since a single exposure from most camera sensors cannot capture the full dynamic range of a high-contrast scene, a common technique is multiple exposure bracketing. This involves capturing a series of images of the same static scene at different exposure levels (e.g., -2, 0, +2 EV). These images are then merged in software to create a single HDR image that contains the detail from all the exposures.39
*   **HDR Formats**: To store this extended range of data, specialized file formats are required. Unlike standard 8-bit integer formats, HDR images are typically stored in 16-bit or 32-bit floating-point formats, such as OpenEXR or Radiance HDR (.hdr). These formats can encode luminance values that far exceed the 1.0 "white" level of SDR, preserving the true radiance values of the original scene.89

#### 10.2 Tone Mapping: From HDR to SDR
An HDR image cannot be viewed directly on a conventional SDR display; its vast dynamic range must be compressed to fit within the limited capabilities of the screen. This compression process is known as Tone Mapping.87 The goal of a tone mapping operator (TMO) is not just to compress the data, but to do so in a way that preserves the perceived detail, contrast, and color appearance of the original HDR scene.91

Tone mapping is fundamentally a perceptual problem, not just a mathematical one. While the core process is a mathematical compression of data, its true goal is perceptual fidelity: to "reproduce the visual appearance of HDR scenes based on human perception".91 This reframes the entire problem. Algorithms are not just fitting curves; they are attempting to model complex aspects of human vision, such as how our eyes adapt to local contrast. This explains the persistent trade-off between the two main categories of tone mapping operators:

*   **Global Operators**: These algorithms apply a single, non-linear curve to every pixel in the image based on global properties like average luminance. A well-known example is the Reinhard operator. They are computationally simple and fast, but because they treat all pixels equally, they often reduce local contrast, which can make the image appear flat.90
*   **Local Operators**: These are more sophisticated, spatially-varying algorithms. The tone curve is adapted for each pixel based on the luminance of its surrounding neighborhood. This allows them to preserve local contrast and detail much more effectively, mimicking how the human eye adapts to different brightness levels within a scene. However, they are more computationally expensive and can introduce visual artifacts like unnatural "halos" around high-contrast edges if not carefully implemented.90

#### 10.3 HDR Standards and Ecosystem (up to 2025)
For HDR content to be delivered to consumers, a standardized ecosystem of formats and display technologies is required. As of 2025, several key standards are prevalent:

*   **HDR10**: The open, royalty-free baseline standard for HDR content. It uses 10-bit color depth and the PQ (Perceptual Quantizer) transfer function. Its primary limitation is the use of static metadata, meaning a single set of tone mapping information is applied to an entire piece of content.94
*   **HDR10+**: An evolution of HDR10, also open and royalty-free, that incorporates dynamic metadata. This allows the tone mapping information to be adjusted on a scene-by-scene or even frame-by-frame basis, enabling a more optimized presentation on displays with different brightness capabilities.94
*   **Dolby Vision**: A proprietary HDR format that also uses dynamic metadata. It supports up to 12-bit color depth and is widely used in high-end consumer electronics and streaming services.
*   **Hybrid Log-Gamma (HLG)**: A standard developed primarily for broadcast television. Its key advantage is that it is a single signal that is backward-compatible: HDR displays can interpret the HLG signal to show an HDR image, while older SDR displays can interpret the same signal to show a standard SDR image, simplifying the broadcast chain.88

The landscape of HDR technology continues to evolve rapidly. Key advancements highlighted at major 2025 industry events like CES and NAB include the emergence of multi-layer "tandem" OLED display technology capable of achieving peak brightness levels approaching 4,000 nits, a significant leap forward in HDR reproduction.96 Additionally, there is growing adoption of HDR10+ GAMING for optimized gaming experiences and the deployment of Advanced HDR by Technicolor for NEXTGEN TV (ATSC 3.0) broadcasting, indicating a broadening of HDR application beyond cinematic content.95

### References
39. [Citation text to be added]
86. [Citation text to be added]
87. [Citation text to be added]
88. [Citation text to be added]
89. [Citation text to be added]
90. [Citation text to be added]
91. [Citation text to be added]
94. [Citation text to be added]
95. [Citation text to be added]
96. [Citation text to be added]
