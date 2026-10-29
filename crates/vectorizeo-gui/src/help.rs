//! Tooltip texts (the page or topic each belongs to in brackets).

// [Image Type]
pub const PHOTO: &str = "Photographs of natural scenes where precise reconstruction is not important. \
    The simplest and fastest mode; tuned for very large images.";
pub const BLENDED: &str = "Bitmaps rasterised from vector art with anti-aliasing (colours blend at shape \
    boundaries). Extracts very small features from low-resolution images; the slowest mode.";
pub const UNBLENDED: &str = "Aliased bitmaps with jagged, pixel-aligned boundaries. Using it on an \
    anti-aliased image (or vice versa) gives very bad results.";

// [Photograph Output Detail]
pub const DETAIL: &str = "Trades fidelity to the bitmap against complexity. High is the most faithful and \
    uses very many shapes; Medium balances shape count and likeness; Low looks vaguely cartoonish.";
// [Anti-Aliased / Aliased Artwork Input Quality]
pub const QUALITY: &str = "Describes the input, not the output. High: crisp and clean, blending at most \
    one pixel wide. Medium: some noise or JPEG artifacts, blending up to two pixels (the safest choice). \
    Low: very noisy or degraded, low-quality JPEGs and scans.";

// [Color Mode]
pub const FIXED: &str = "Only these colours are used. Specifying the palette reduces the effect of noise \
    and can give stylised results. Two colours suit black-and-white images.";
pub const FLEXIBLE: &str = "The colours are found while segmenting. Usually many more colours than a \
    fixed palette, and able to tell smaller differences apart.";
pub const CUSTOM: &str = "Edit the palette by hand: add, remove or change colours.";

// [Palette Selection]
pub const SENSITIVITY: &str = "The more sensitive, the better small colour differences are told apart, \
    but noisy pixels are more likely to be taken for real colours. Medium is a safe choice.\n\
    High: 16 histogram bins, 6-connected peaks · Medium: 16 bins, 26-connected · Low: 8 bins, 26-connected.";
pub const PALETTE_AA: &str = "Whether the image is anti-aliased (blending at the shape boundaries). \
    Zoom in on an edge to check.";

// [Image Segmentation]
pub const COMPLEXITY: &str = "Fidelity to the bitmap against the number of segments: more or fewer \
    segments. Each segment becomes one vector shape.";
pub const MIN_PIXELS: &str = "Segments with fewer pixels are merged into a neighbour. Anti-aliasing \
    artifact rejection is usually the better tool.";
pub const AA_REJECTION: &str = "Removes segments that are likely artifacts of anti-aliasing along the \
    boundaries between shapes. Important for anti-aliased images.";
pub const CLUSTER: &str = "With the flexible palette each shape's colour is estimated on its own; this \
    forces segments with similar colours to share one colour.";

// [Path Smoothing]
pub const SMOOTHNESS: &str = "Less smoothing follows the contours closely (clean images); more smoothing \
    ignores small details for flatter, smoother curves (noisy images).";
pub const CORNERS: &str = "Explicitly look for sharp corners. Turned off, corners can still emerge, \
    they are only less likely.";
pub const AA_ENGINE: &str = "Model the anti-aliasing process while smoothing. Turn on only if edges \
    blend between colours.";
pub const NODES: &str = "How many nodes the paths use. Changing this re-runs less than the other \
    settings: tune smoothness with many nodes first, then reduce them.";

// [Transparency]
pub const FLATTEN_WHITE: &str = "Flatten the image onto white: by far the most common choice for images \
    that need flattening (edges blended with white).";
pub const FLATTEN_COLOR: &str = "Flatten onto the colour the edges were anti-aliased to.";
pub const LEAVE_TRANSPARENT: &str = "Keep the alpha channel: right when edges blend to transparent, and \
    for aliased images. Not for anti-aliased images with a 1-bit (GIF) mask.";

// Export options.
pub const STACKED: &str = "Shapes are placed on top of each other.";
pub const CUTOUTS: &str = "Shapes sit in cut-outs (holes) of the shapes below.";
pub const GROUPED: &str = "Cut-outs, with the shapes grouped by colour.";
pub const STROKE: &str = "Outline every shape with a hairline in its fill colour (hides seams in some viewers).";
