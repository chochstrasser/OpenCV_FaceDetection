
// load image from file
int Load_Image();
int Image_Detect();

// Start Video Session
int Load_Video();

// Smoothing Images
int HomogeneousBlur();
int GaussianBlur();
int MedianBlur();
int BilateralFilter();

//
int Eroding();
int Dilating();
int Saving_Video_Capture();

// Morphology Transformation
int Morph_Open();
int Morph_Close();
int Morph_Gradient();
int Morph_TopHat();
int Morph_BlackHat();

// Object Detection
int DetectFace();

int FisherFace();
int LBPH_Face();

// Re-Mapping
int Remap();
int Remap_Video();
