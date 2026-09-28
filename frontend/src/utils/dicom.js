import daikon from 'daikon';

export async function parseDicom(file) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const data = new DataView(arrayBuffer);
    
    // Parse DICOM
    const image = daikon.Series.parseImage(data);
    if (!image) {
      throw new Error("Invalid DICOM file");
    }

    // Get pixel data
    const rawData = image.getInterpretedData();
    const cols = image.getCols();
    const rows = image.getRows();

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(cols, rows);

    // Get min/max for normalization
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < rawData.length; i++) {
      if (rawData[i] < min) min = rawData[i];
      if (rawData[i] > max) max = rawData[i];
    }

    const range = max - min;
    const isMonochrome1 = image.getPhotometricInterpretation() === 'MONOCHROME1';

    for (let i = 0; i < rawData.length; i++) {
      let val = ((rawData[i] - min) / range) * 255;
      if (isMonochrome1) {
        val = 255 - val;
      }
      
      const idx = i * 4;
      imgData.data[idx] = val;     // R
      imgData.data[idx + 1] = val; // G
      imgData.data[idx + 2] = val; // B
      imgData.data[idx + 3] = 255; // A
    }

    ctx.putImageData(imgData, 0, 0);

    // Convert to Data URL and Blob
    const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
    
    // Convert Data URL to Blob
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    const jpgFile = new File([blob], file.name.replace('.dcm', '.jpg'), { type: 'image/jpeg' });
    
    return {
      file: jpgFile,
      previewUrl: dataUrl
    };
  } catch (err) {
    console.error("DICOM Parse Error:", err);
    throw new Error("Failed to parse DICOM image. Make sure it contains pixel data.");
  }
}
