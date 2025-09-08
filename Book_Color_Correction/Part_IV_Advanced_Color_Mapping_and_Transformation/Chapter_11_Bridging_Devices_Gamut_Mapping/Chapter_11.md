### Chapter 11: Bridging Devices: Gamut Mapping
Just as tone mapping bridges the gap in dynamic range between devices, gamut mapping addresses the inevitable discrepancies in their color reproduction capabilities. This process is essential for translating color information from a wide-gamut source to a destination device with a more limited color range.

#### 11.1 The Gamut Mismatch Problem
A device's gamut is the complete range of colors it can produce or capture.23 In nearly every professional workflow, a gamut mismatch occurs: the gamut of the source device (like a digital camera capturing in ProPhoto RGB) is significantly larger than the gamut of the destination device (like a monitor displaying sRGB or a printer using CMYK).99 This means there are colors in the source image that the destination device is physically incapable of reproducing. These are known as out-of-gamut colors.68

#### 11.2 Gamut Mapping Algorithms (GMAs) and Rendering Intents
A Gamut Mapping Algorithm (GMA) is the strategy used to handle these out-of-gamut colors, remapping them to the closest available colors within the destination gamut.100 The International Color Consortium (ICC) has standardized four primary strategies for this process, known as Rendering Intents.68

The necessity of gamut mapping arises from the physical impossibility of a smaller gamut reproducing all the colors of a larger one. This means the color management system must alter some colors—it must, in a sense, "lie." The four rendering intents are not just different algorithms; they represent four distinct philosophical approaches to how to lie gracefully and effectively, depending on the goal of the reproduction.

*   **Perceptual**: This intent aims to preserve the overall visual relationship between all the colors in the source image. It does this by uniformly compressing the entire source gamut to fit inside the destination gamut. While this preserves smooth gradients and the relative appearance of colors, it means that even colors that were originally in-gamut will be slightly shifted.99
    *   **Philosophy**: "Lie a little about every color to maintain the overall visual harmony."
    *   **Best Use**: Photographic images, where the subtle relationships between colors are more important than the precise accuracy of any single color.
*   **Relative Colorimetric**: This intent prioritizes the accuracy of in-gamut colors. It maps colors that are within both gamuts exactly, and "clips" any out-of-gamut colors to the nearest reproducible color on the boundary of the destination gamut. It also scales the white point of the source to match the white point of the destination.99
    *   **Philosophy**: "Tell the truth about all the colors we can, and for the others, find the closest possible substitute."
    *   **Best Use**: Vector graphics and logos, where maintaining the exact values of specific brand colors (if they are in-gamut) is critical. The potential loss of detail in saturated, out-of-gamut areas is an acceptable trade-off.
*   **Absolute Colorimetric**: This intent is identical to Relative Colorimetric in its handling of in-gamut and out-of-gamut colors, with one crucial difference: it does not adjust the white point. It preserves the absolute white of the source, which means it will simulate the color of the paper or substrate of the source profile.99
    *   **Philosophy**: "Tell the absolute, unvarnished truth, even if it means showing the yellowish 'white' of the paper."
    *   **Best Use**: Hard proofing. It is used to accurately simulate on one device (e.g., a proofer) how an image will look when printed on another device with a different white point (e.g., the final press).
*   **Saturation**: This intent's primary goal is to preserve the vividness and saturation of colors, often at the expense of accuracy in hue and lightness. It maps saturated source colors to saturated destination colors.99
    *   **Philosophy**: "Lie in whatever way creates the most visual impact."
    *   **Best Use**: Business graphics like charts and graphs, where the goal is to create visually distinct and impactful colors rather than photorealistic reproduction.

#### 11.3 Advanced Gamut Mapping Strategies
Beyond the standard rendering intents, research continues into more sophisticated GMAs. The common objectives of these algorithms include preserving the gray axis to prevent color casts in neutral tones, minimizing hue shifts (as humans are very sensitive to them), and maximizing the use of the destination gamut's saturation potential.102 Techniques such as gamut alignment aim to more intelligently warp the source gamut to better fit the shape of the destination gamut, rather than performing a simple linear compression, in order to produce more vibrant and faithful results.103

### References
23. [Citation text to be added]
68. [Citation text to be added]
99. [Citation text to be added]
100. [Citation text to be added]
102. [Citation text to be added]
103. [Citation text to be added]
