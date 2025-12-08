
import ee
import argparse
from pathlib import Path
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional, Callable
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_gee(project_id: Optional[str] = None):
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project=project_id)
        logger.info("Google Earth Engine initialized successfully.")
    except Exception as e:
        logger.warning(f"GEE initialization failed: {e}")
        logger.info("Trying to authenticate...")
        try:
            ee.Authenticate()
            ee.Initialize(project=project_id)
            logger.info("Google Earth Engine authenticated and initialized successfully.")
        except Exception as e2:
            logger.error(f"Authentication failed: {e2}")
            logger.error("Please run `earthengine authenticate` in your terminal.")
            exit(1)

# --- Processing Functions ---

def process_sentinel2(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    # Select specific bands (B2, B3, B4, B8, B11) and scale
    return image.updateMask(mask).divide(10000).select(['B2', 'B3', 'B4', 'B8', 'B11'])

def process_sentinel1(image):
    # Select VV and VH
    return image.select(['VV', 'VH'])

def process_landsat(image):
    # Select bands B2-B7 (Blue, Green, Red, NIR, SWIR1, SWIR2)
    return image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])

def process_gedi(image):
    # Select relative height metrics
    return image.select(['rh98', 'rh100', 'rh50', 'rh25', 'rh75'])

def process_era5(image):
    return image.select(['total_precipitation_sum', 'temperature_2m', 'dewpoint_temperature_2m'])

def process_glo30(image):
    return image.select(['DEM'])

def process_nlcd(image):
    return image.select(['landcover'])


# --- Configuration ---

class SourceConfig:
    def __init__(self, collection_name: str, bands: List[str], process_func: Callable, static: bool = False):
        self.collection_name = collection_name
        self.bands = bands
        self.process_func = process_func
        self.static = static

SOURCE_CONFIGS = {
    "sentinel2": SourceConfig(
        "COPERNICUS/S2_HARMONIZED", 
        ['B2', 'B3', 'B4', 'B8', 'B11'], 
        process_sentinel2
    ),
    "sentinel1": SourceConfig(
        "COPERNICUS/S1_GRD",
        ['VV', 'VH'],
        process_sentinel1
    ),
    "landsat8": SourceConfig(
        "LANDSAT/LC08/C02/T1_TOA",
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        process_landsat
    ),
    "landsat9": SourceConfig(
        "LANDSAT/LC09/C02/T1_TOA",
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        process_landsat
    ),
    "gedi": SourceConfig(
        "LARSE/GEDI/GEDI02_A_002_MONTHLY",
        ['rh98', 'rh100', 'rh50', 'rh25', 'rh75'],
        process_gedi
    ),
    "era5": SourceConfig(
        "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        ['total_precipitation_sum', 'temperature_2m', 'dewpoint_temperature_2m'],
        process_era5
    ),
    "glo30": SourceConfig(
        "COPERNICUS/DEM/GLO30",
        ['DEM'],
        process_glo30,
        static=True
    ),
    "nlcd": SourceConfig(
        "USGS/NLCD_RELEASES/2019_REL/NLCD",
        ['landcover'],
        process_nlcd,
        static=True
    )
}

def get_collection(source_name: str, start_date: str, end_date: str):
    config = SOURCE_CONFIGS.get(source_name)
    if not config:
        raise ValueError(f"Unknown source: {source_name}")
    
    collection = ee.ImageCollection(config.collection_name)
    
    if not config.static:
        collection = collection.filterDate(start_date, end_date)
        
    if source_name == "sentinel1":
        # Specific filters for S1
        collection = (collection
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
        )
    elif source_name == "sentinel2":
        collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
    return collection.map(config.process_func), config.bands, config.static


def download_single_point(
    idx: int,
    lat: float, 
    lon: float, 
    collection: ee.ImageCollection, 
    source_name: str,
    bands: List[str],
    is_static: bool,
    output_dir: Path, 
    patch_size: int = 128, 
    scale: int = 10
) -> bool:
    """Download data for a single point. Returns True if successful."""
    try:
        point = ee.Geometry.Point([lon, lat])
        buffer_dist = (patch_size * scale) / 2
        roi = point.buffer(buffer_dist).bounds()
        
        filtered = collection.filterBounds(point)
        
        # Determine number of images to fetch
        if is_static:
            limit = 1
        else:
            # Check size, but cap it to avoid huge unexpected downloads
            # For dynamic collections, we take up to 50 recent images
            # Optimally we might want to sort by time
            count = filtered.size().getInfo()
            if count == 0:
                return False
            limit = min(count, 50)
            
        image_list = filtered.toList(limit)
        
        source_data = []
        timestamps_list = []
        
        # Pre-fetch list size to be safe
        actual_limit = image_list.size().getInfo()
        
        for k in range(actual_limit):
            img = ee.Image(image_list.get(k))
            ts = 0
            if not is_static:
                try:
                    ts = img.date().millis().getInfo()
                except:
                    pass
            
            # Robust download/sampling
            try:
                # Retries for network robustness
                max_retries = 3
                sample_dict = None
                for attempt in range(max_retries):
                    try:
                        sample = img.sampleRectangle(region=roi, defaultValue=0)
                        sample_dict = sample.getInfo()
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        time.sleep(1 * (attempt + 1)) # Backoff

                properties = sample_dict.get('properties', {})
                
                single_image_data = []
                valid_bands = True
                
                for b in bands:
                    if b not in properties:
                        valid_bands = False
                        break
                    
                    arr = np.array(properties[b])
                    
                    # Ensure correct shape (center crop if needed)
                    h, w = arr.shape
                    if h < patch_size or w < patch_size:
                        valid_bands = False
                        break
                        
                    diff_h = (h - patch_size) // 2
                    diff_w = (w - patch_size) // 2
                    arr = arr[diff_h:diff_h+patch_size, diff_w:diff_w+patch_size]
                    single_image_data.append(arr)
                
                if valid_bands:
                    img_np = np.stack(single_image_data, axis=-1)
                    source_data.append(img_np)
                    timestamps_list.append(ts)
            
            except Exception as e:
                # Log usage warning only if verbose
                # logger.warning(f"Failed to process image {k} for point {idx}: {e}")
                continue

        if source_data:
            source_data_arr = np.stack(source_data, axis=0)
            timestamps_arr = np.array(timestamps_list)
            
            out_file = output_dir / f"point_{idx}_{source_name}.npz"
            save_dict = {
                source_name: source_data_arr,
                f"ts_{source_name}": timestamps_arr
            }
            np.savez_compressed(out_file, **save_dict)
            return True
            
    except Exception as e:
        logger.error(f"Error downloading point {idx} ({lat}, {lon}): {e}")
        return False
        
    return False

def download_chips(
    coords: List[Tuple[float, float]],
    start_date: str,
    end_date: str,
    output_dir: Path,
    source_name: str,
    patch_size: int = 128,
    scale: int = 10,
    max_workers: int = 4
):
    """
    Downloads chips for a given source and coordinates using parallelism.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    collection, bands, is_static = get_collection(source_name, start_date, end_date)
    
    logger.info(f"Starting download for {source_name}. Total points: {len(coords)}")
    
    success_count = 0
    
    # Use ThreadPoolExecutor for concurrent downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (lat, lon) in enumerate(coords):
            futures.append(
                executor.submit(
                    download_single_point, 
                    i, lat, lon, collection, source_name, bands, is_static, output_dir, patch_size, scale
                )
            )
            
        # Progress bar
        with tqdm(total=len(coords), desc=f"Downloading {source_name}") as pbar:
            for future in concurrent.futures.as_completed(futures):
                if future.result():
                    success_count += 1
                pbar.update(1)
                
    logger.info(f"Finished {source_name}. Successfully downloaded {success_count}/{len(coords)} points.")

def main():
    parser = argparse.ArgumentParser(description="Download data from GEE.")
    parser.add_argument("--project", type=str, help="Google Cloud Project ID", default=None)
    parser.add_argument("--output", type=str, default="data/train_npz", help="Output directory")
    parser.add_argument("--samples", type=int, default=1, help="Samples count")
    parser.add_argument("--sources", type=str, default="sentinel2", help="Comma-separated sources: sentinel2,sentinel1,landsat8,gedi,era5,glo30,nlcd")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()
    
    initialize_gee(args.project)
    
    # Generate points
    # Default seeds
    locations = [
        (37.7749, -122.4194),
        (40.7128, -74.0060),
    ]
    if args.samples > len(locations):
        for _ in range(args.samples - len(locations)):
            lat = np.random.uniform(-50, 50)
            lon = np.random.uniform(-180, 180)
            locations.append((lat, lon))
            
    locations = locations[:args.samples]
    output_dir = Path(args.output)
    
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    source_list = args.sources.split(',')
    
    for source in source_list:
        try:
            download_chips(
                coords=locations,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir,
                source_name=source.strip(),
                patch_size=128,
                scale=10,
                max_workers=args.workers
            )
        except Exception as e:
            logger.error(f"Failed to download {source}: {e}")

if __name__ == "__main__":
    main()
