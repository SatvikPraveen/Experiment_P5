from pathlib import Path
import SimpleITK as sitk
import numpy as np
import imageio.v3 as iio
import astra
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pdb
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


def visualize(ct_slice: np.ndarray):
    
    ct_slice = ct_slice.astype(np.float32, copy=False)
    lo = np.percentile(ct_slice, 2)
    hi = np.percentile(ct_slice, 97)
    #lo, hi = [-3024.0, 1000.0]
    gt_clipped = np.clip(ct_slice, lo, hi).astype(np.float32, copy=False)
    gt_clipped = 0.2 * (1.0 + gt_clipped / 1000.0)
    vmin, vmax = float(np.min(gt_clipped)), float(np.max(gt_clipped))
    if vmax > vmin:
        gt_clipped =  (gt_clipped - vmin)  / (vmax - vmin)
        v_u8 = (gt_clipped * 255.0 + 0.5).astype(np.uint8)
        iio.imwrite("ct.png", v_u8)
    # print("done")
    

#Need to work on this to convert to different configs later
# need to do the sampling using astra
def convert_hu_to_mu(ct_slice):
    """Convert HU to linear attenuation coefficient (mu).
    mu_water ~ 0.02 mm^-1 at typical CT energies.
    """
    ct_slice = ct_slice.astype(np.float32, copy=False)
    # HU to mu: mu = mu_water * (1 + HU/1000)
    mu = 0.02 * (1.0 + ct_slice / 1000.0)
    mu = np.clip(mu, 0, None)  # mu cannot be negative
    return mu

#Need to convert to sinogram and visualize
def convert_sinogram(ct_slice, dx, dy, dz):
    
    #det count should be width of a pixel
    #dx_mm which is det_spacing is dx_mm
    
    # Convert HU to attenuation coefficients first
    mu_slice = convert_hu_to_mu(ct_slice)
    
    H, W = mu_slice.shape
    
    DSO = 1000  
    ODD = 600  
    angles_deg = np.arange(0, 360, 0.25, dtype=np.float32)
    angles = np.deg2rad(angles_deg)  # ASTRA expects radians
    
    # generate params for the second part
    vol_geom = astra.create_vol_geom( H, W,
        -W * dx / 2.0,  W* dx/ 2.0,   # x_min, x_max
        -H * dy / 2.0,  H * dy / 2.0    # y_min, y_max
    )
        
    # Detector should cover the full object diagonal
    det_count = int(np.ceil(np.sqrt(H**2 + W**2) * 1.5))
    det_spacing = dx  
    
    proj_geom = astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, ODD)
    
    projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    slice2d = np.ascontiguousarray(mu_slice, dtype=np.float32)  # (H, W)
    sid = astra.data2d.create('-vol', vol_geom, slice2d)
    sino_id, sino = astra.create_sino(sid, projector_id) 
    
    # Cleanup ASTRA resources
    astra.data2d.delete(sino_id)
    astra.data2d.delete(sid)
    astra.projector.delete(projector_id)
    print(sino.shape)
    # Display the array as a grayscale image
    # plt.imshow(sino, cmap='gray')
    # plt.show()
    projection(sino)
    exit(0)
    return sino  # Return the sinogram
    
def load_series_from(path):
    #set path of the Dicom series
    series_dir = Path(path)
    
    
    sitk_reader = sitk.ImageSeriesReader()
    file_names = sitk_reader.GetGDCMSeriesFileNames(str(series_dir))
    sitk_reader.SetFileNames(file_names)
    img = sitk_reader.Execute()
    
    vol_zyx = sitk.GetArrayFromImage(img)  
    spacing_x, spacing_y, spacing_z = img.GetSpacing()
    visualize(vol_zyx[0])
    return vol_zyx, (spacing_x, spacing_y, spacing_z)
    
    #get the volumes and the spacings


def projection(img_array):
    sino_AD = np.asarray(img_array, dtype=np.float32)

    # Sinogram dimensions: (num_angles, num_detectors)
    num_angles = sino_AD.shape[0]
    num_detectors = sino_AD.shape[1]

    DSO = 1000  
    ODD = 600  

    angles_deg = np.arange(0, 360, 0.25, dtype=np.float32)  # Match convert_sinogram
    angles = np.deg2rad(angles_deg)  # ASTRA expects radians

    spacing_xyz = (0.664062, 0.664062, 2.5000000984848483)

    # Reconstruction volume size - use original image dimensions (before diagonal expansion)
    # det_count in convert_sinogram = ceil(sqrt(H^2 + W^2) * 1.5), so reverse it
    original_size = int(num_detectors / (np.sqrt(2) * 1.5))
    H = original_size
    W = original_size

    dx = spacing_xyz[0]
    dy = spacing_xyz[1]

    # generate params for the second part
    vol_geom = astra.create_vol_geom(H, W,
        -W * dx / 2.0,  W * dx / 2.0,   # x_min, x_max
        -H * dy / 2.0,  H * dy / 2.0    # y_min, y_max
    )
        
    # Detector count must match sinogram width
    det_count = num_detectors
    det_spacing = dx  

    proj_geom = astra.create_proj_geom('fanflat', det_spacing, det_count, angles, DSO, ODD)

    projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    sinogram_id = astra.data2d.create('-sino', proj_geom, sino_AD)
    recon_id = astra.data2d.create('-vol', vol_geom)

    cfg = astra.astra_dict("FBP_CUDA")  # Use GPU FBP for fan-beam
    cfg["ProjectionDataId"] = sinogram_id
    cfg["ReconstructionDataId"] = recon_id
    cfg["ProjectorId"] = projector_id
    cfg["FilterType"] = "Ram-Lak"

    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id)
        result = astra.data2d.get(recon_id).astype(np.float32, copy=False)
    except:
        print("Not done")
    finally:
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(recon_id)

    arr = result

    # Debug: print value range
    print(f"Result range: min={arr.min()}, max={arr.max()}")

    # Create a circular mask to focus on the center (ignore dark corners)
    center_y, center_x = arr.shape[0] // 2, arr.shape[1] // 2
    Y, X = np.ogrid[:arr.shape[0], :arr.shape[1]]
    radius = min(center_x, center_y) * 0.85  # 85% of the radius
    mask = ((X - center_x)**2 + (Y - center_y)**2) <= radius**2

    # Use percentiles from inside the circular region only
    arr_masked = arr[mask]
    lo = np.percentile(arr_masked, 2)   # Less aggressive clipping
    hi = np.percentile(arr_masked, 98)
    print(f"Masked percentile range: lo={lo}, hi={hi}")

    arr_clipped = np.clip(arr, lo, hi)

    # Normalize to [0, 1]
    if hi > lo:
        arr_normalized = (arr_clipped - lo) / (hi - lo)
    else:
        arr_normalized = arr_clipped

    # Apply CLAHE for local contrast enhancement (reduced effect)

    # Gamma correction for brightness
    gamma = 0.8  # Closer to 1 = less brightening
    arr_bright = np.power(arr_normalized, gamma)

    arr_to_write = (arr_normalized * 255.0).astype(np.uint8)
    iio.imwrite(f"output.png", arr_to_write)
    print(f"Saved output.png")


#correct so far 
#vol_zyx , spacing = load_series_from("data/Dataset")
#sinograms = [convert_sinogram(slice, spacing[0], spacing[1], spacing[2]) for slice in vol_zyx]

class dicom_dataset(Dataset):
    def __init__(self, data_path="data/Dataset", trajec_max_len=50, start_n=0, n_span=None):
        vol_zyx, spacing = load_series_from(data_path)
        print(spacing)
        # Convert all slices to sinograms
        self.sinograms = []
        for ind in range(20):
            sino = convert_sinogram(vol_zyx[ind], spacing[0], spacing[1], spacing[2])
            
            # Normalize to [-1, 1]
            sino_min, sino_max = sino.min(), sino.max()
            if sino_max > sino_min:
                sino = 2.0 * (sino - sino_min) / (sino_max - sino_min) - 1.0
                        
            # NumPy -> Torch
            sino = torch.from_numpy(sino).float()
            
            sino = sino.unsqueeze(0).unsqueeze(0)

            # Resize
            sino = F.interpolate(   
                sino,
                size=(1024, 1024),
                mode="nearest"   # correct for sinograms
            )

            # Remove batch + channel dims: [512, 512]
            sino = sino.squeeze(0).squeeze(0)

            # Torch -> NumPy
            sino = sino.detach().cpu().numpy()

            #print(sino.shape)  # (512, 512)
            
            #np.save("sino.npy", sino)
            self.sinograms.append(sino)
        # Stack to [num_sinograms, H, W] then add two dimensions -> [num_sinograms, 1, 1, H, W]
        self.sinograms_torch = torch.from_numpy(np.stack(self.sinograms, axis=0)).unsqueeze(1).unsqueeze(1)

    def __len__(self):
        return len(self.sinograms)
            
    def __getitem__(self, index):
        return self.sinograms_torch[index]



if __name__ == '__main__':
	dset = dicom_dataset()

	print("here")
	dloader = DataLoader(dataset=dset, batch_size = 5,shuffle = True)
	print("here")
	for iteration, batch in enumerate(dloader):
		print(batch.shape)




