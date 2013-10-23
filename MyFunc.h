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

// Croping face and saving face
void CropFace();
void SaveFace();

// Video Stuff
int Eroding();
int Dilating();
int Saving_Video_Capture();

// Morphology Transformation
int Morph_Open();
int Morph_Close();
int Morph_Gradient();
int Morph_TopHat();
int Morph_BlackHat();

// Re-Mapping
int Remap();
int Remap_Video();
