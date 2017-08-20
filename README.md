## Data analysis package

This package contains three modules:
1. file_loader.py
2. image_processor.py
3. hkl_calculator.py

_Note classes are listed in the format:_
- _Parent_
  * _Child 1_
  * _Child 2_

### file_loader.py

_I/O module._

This module creates a File class that can fetch data and/or plot data.

Available classes are following
- File
  * MCPDataFile
  * TESConfigFile
  * SPECParamFile
  * ScanFileAPS29
  * SCanFileSSRL13

### image_processor.py

_Image processor module._

This module creates 2D Image class that contains data plotting, peak searching
and plotting, line cutting (which returns 1D Line class).

This module creates 1D Line class that contains plotting, fitting including
scipy.curve_fit, lmfit and kernel density estimation.

Available classes are one of following
- Image2D
- Line1D
  * LineCut
  * Line1DAPS29
  * Line1DSSRL13
  * Hist1DTES

### hkl_calculator.py

_hkl mapping module._

This module creates AreaDetector class that can calculate hkl for each pixel
given all angles in 4-circle diffractometer.

This module is accurate with given orientation matrix UB calculated from spec;
however, with respect to building orientation matrix UB, there's algorithmic
error between my calculation and spec ;(.

Available classes are one of following
- AreaDetector
  * MCP
  * TES
