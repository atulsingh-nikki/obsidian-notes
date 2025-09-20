### Chapter 9: Color Standards for Broadcast and Print
While workflows like ACES provide a universal framework, final delivery of content must conform to specific, long-established standards tailored to the target medium, whether it be a television broadcast or a printed page.

#### 9.1 Rec. 709 for Broadcast Television
The foundational color standard for all high-definition (HD) television, from broadcast to Blu-ray, is ITU-R Recommendation BT.709, commonly known as Rec. 709.30 This standard precisely defines the parameters for HD video, including:

*   **Resolution**: 1920x1080 pixels.
*   **Aspect Ratio**: 16:9.
*   **Color Primaries and White Point**: It specifies the exact chromaticity coordinates for the red, green, and blue primaries and a D65 white point. The Rec. 709 gamut is nearly identical to the sRGB gamut.30
*   **Transfer Function (Gamma)**: It defines a specific non-linear transfer function (often approximated as a 2.4 gamma curve) that dictates how the digital code values map to light output on a display calibrated for a dim viewing environment.80

For content creators, the workflow involves taking footage captured in a camera's native Log format (which has a flat, low-contrast look but retains maximum dynamic range) and transforming it into the Rec. 709 color space for broadcast delivery. This is often done using a technical Look-Up Table (LUT) that correctly maps the camera's specific Log curve and gamut to the Rec. 709 standard.80

**Challenges in Live Broadcast**: Live production presents unique and formidable color challenges. Multiple cameras must be matched in real-time under lighting conditions that can change unpredictably. Furthermore, the increasing use of large LED video walls as backdrops introduces another layer of complexity, as these displays often have color gamuts and rendering characteristics that differ significantly from broadcast cameras, leading to color rendition issues, particularly in yellows and cyans, that require sophisticated real-time correction.82

#### 9.2 Color Management for Digital Printing
The workflow for professional digital printing is a highly controlled process designed to achieve predictable and repeatable results. The modern prepress workflow, as of 2025, can be summarized by the "6 C's" of color management.84

*   **Consistency**: Establishing a stable production environment. This means ensuring the printing press is operating correctly, substrates and inks are consistent from batch to batch, and environmental factors like humidity are controlled.
*   **Calibration**: Aligning all devices to a known target. This involves linearizing the printer's output and establishing a neutral gray balance, often using methodologies like G7.
*   **Characterization**: Creating an accurate ICC profile for each unique combination of printer, ink, and substrate. This is done by printing a standardized color target and measuring the printed patches with a spectrophotometer to build a profile that precisely describes that system's color reproduction capabilities.
*   **Conversion**: Using the created ICC profiles to translate color across different devices and color spaces. This is the crucial step where RGB data from a design file is converted to the CMYK color space of the printer. This conversion is guided by a chosen rendering intent (see Chapter 11) to handle out-of-gamut colors intelligently.
*   **Control**: Continuously monitoring quality to ensure long-term accuracy. This involves regularly printing control strips and measuring them to verify that the press is still operating within the specified tolerances (e.g., a certain Delta E value).
*   **Conformance**: Adhering to industry standards and client expectations, ensuring that the final product meets the required specifications for brand colors and overall appearance.

The modern color management landscape is defined by a fundamental tension. On one side are the unifying forces of standards like Rec. 709 and G7, which aim to create a universal language for color reproduction.30 On the other side is the fragmenting force of market competition, which has led to a proliferation of proprietary camera Log formats (S-Log, C-Log, V-Log, etc.).85 Each manufacturer develops its own unique "color science," creating a multitude of different starting points that must all be wrangled into a common framework. This conflict explains the rise of tools designed specifically to bridge this gap: technical LUTs whose sole purpose is to convert a specific Log curve to a standard like Rec. 709 80, and automated features in editing software that attempt to auto-detect and manage these disparate sources.85 The daily work of a modern colorist is largely about mediating this battle between standardization and proliferation.

### References
30. [Citation text to be added]
80. [Citation text to be added]
82. [Citation text to be added]
84. [Citation text to be added]
85. [Citation text to be added]
