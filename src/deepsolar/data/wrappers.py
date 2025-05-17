# -*- coding: utf-8 -*
import io
import multiprocessing as mp
import os
import urllib.request as ur
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import googlemaps
import numpy as np
import pandas as pd
from PIL import Image
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    SentinelHubCatalog,
    SentinelHubDownloadClient,
    SHConfig,
)

import deepsolar.data.utils as utils


class GoogleMapsWebDownloader(object):
    def __init__(self):
        self.tiles = None

    def download(self, top_left, right_bottom, folder, **kwargs):
        if not os.path.exists(folder):
            os.mkdir(folder)

        urls, xy = self._get_urls(top_left, right_bottom, kwargs.get("zoom"), kwargs.get("style"))

        results = self._download(urls, folder, *xy, kwargs.get("format"))
        tiles = [img for row in results for img in row]

    def _download(self, urls, folder, len_x, len_y, fformat):
        path = Path(folder)
        path.mkdir(exist_ok=True, parents=True)

        per_process = len(urls) // mp.cpu_count()

        urls_and_names = [(url, rf"{folder}\tile_{i // len_x}_{i % len_y}.{fformat}") for i, url in enumerate(urls)]
        split_urls = [urls_and_names[i : i + per_process] for i in range(0, len(urls_and_names), per_process)]
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
            results = list(pool.map(self._download_tiles, split_urls))
        return results

    def _download_tiles(self, urls_and_names):
        with ThreadPoolExecutor(max_workers=32) as pool:
            byte_images = list(pool.map(lambda v: self._request(v[0], v[1]), urls_and_names))
        return byte_images

    def _get_urls(self, top_left, right_bottom, zoom, style):
        pos1x, pos1y, pos2x, pos2y = utils.latlon2px(*top_left, *right_bottom, zoom)
        len_x, len_y = utils.get_region_size(pos1x, pos1y, pos2x, pos2y)

        return [
            self.get_url(i, j, zoom, style) for j in range(pos1y, pos1y + len_y) for i in range(pos1x, pos1x + len_x)
        ], (len_x, len_y)

    def _request(self, url, name):
        _HEADERS = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 Edg/88.0.705.68"
        }
        header = ur.Request(url, headers=_HEADERS)
        # err = 0
        while True:
            try:
                data = ur.urlopen(header).read()
                self._save_bytes(data, name)
                return data
            except Exception:
                # raise Exception(f"Bad network link: {e}")
                pass

    @staticmethod
    def get_url(x, y, z, style):
        return f"http://mts0.googleapis.com/vt?lyrs={style}&x={x}&y={y}&z={z}"

    @staticmethod
    def _save_bytes(response, output):
        with open(output, "wb") as f:
            for x in response:
                f.write(x)

    def merge(self, filename):
        self._merge_and_save(filename)

    def _merge_and_save(self, filename):
        len_xy = int(np.rint(np.sqrt(len(self.tiles))))
        merged_pic = self._merge_tiles(self.tiles, len_xy, len_xy)
        merged_pic = merged_pic.convert("RGB")
        merged_pic.save(filename)

    @staticmethod
    def _merge_tiles(tiles, len_x, len_y):
        merged_pic = Image.new("RGBA", (len_x * 256, len_y * 256))
        for i, tile in enumerate(tiles):
            tile_img = Image.open(io.BytesIO(tile))
            y, x = i // len_x, i % len_x
            merged_pic.paste(tile_img, (x * 256, y * 256))

        return merged_pic


class GoogleMapsAPIDownloader(object):
    def __init__(self, key):
        self.api = googlemaps.Client(key=key)

    def _request(self, **kwargs):
        return self.api.static_map(**kwargs)

    def download_tile(self, filename, split, **kwargs):
        response = self._request(**kwargs)
        self._save_bytes(response, filename)
        if split:
            self.split_tile(filename)

    def download_grid(self, centers, folder, split=False, **kwargs):
        path = Path(folder)
        path.mkdir(exist_ok=True, parents=True)

        for i, row in enumerate(centers):
            for j, center in enumerate(row):
                filename = f"{folder}/{path.stem}_{j}_{i}.{kwargs.get('format')}"
                self.download_tile(filename=filename, center=center, split=split, **kwargs)

    def parallel_download_grid(self, centers, folder, split=False, **kwargs):
        path = Path(folder)
        path.mkdir(exist_ok=True, parents=True)

        map_params = self._gen_parallel_config(centers, folder, split, **kwargs)

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
            results = list(pool.map(self._download_tiles, map_params))

    def _gen_parallel_config(self, centers, folder, split, **kwargs):
        path = Path(folder)
        map_params = self._gen_map_params(split=split, **kwargs)
        init_args = [
            (center, f"{folder}/{path.stem}_{j}_{i}.{kwargs.get('format')}", map_params)
            for i, row in enumerate(centers)
            for j, center in enumerate(row)
        ]

        per_process = len(init_args) // mp.cpu_count() or 1
        return [init_args[i : i + per_process] for i in range(0, len(init_args), per_process)]

    def _download_tiles(self, centers_and_files):
        with ThreadPoolExecutor(max_workers=32) as pool:
            tiles = list(pool.map(lambda v: self.download_tile(center=v[0], filename=v[1], **v[2]), centers_and_files))
        return tiles

    @staticmethod
    def _gen_map_params(**kwargs):
        return {k: v for k, v in kwargs.items()}

    @staticmethod
    def _save_bytes(response, output):
        with open(output, "wb") as f:
            for x in response:
                f.write(x)

    @staticmethod
    def split_tile(image, size=256):
        image = Path(image)
        path = image.parent / "split"
        path.mkdir(parents=True, exist_ok=True)

        img = np.asarray(Image.open(image).convert("RGB"))
        M, N, *_ = img.shape
        idx = M // size

        for i, row in enumerate(range(0, M, size)):
            for j, col in enumerate(range(0, N, size)):
                tile = np.asarray(Image.new("RGB", (size, size)))
                tile[:, :, :] = img[row : row + size, col : col + size, :]
                Image.fromarray(tile).save(f"{path}/{image.stem}_{(i * idx) + j}{image.suffix}")


class Sentinel2Downloader(object):
    def __init__(self):
        """Initialize with Sentinel Hub credentials.        """
        self.config = SHConfig()
        self.config.sh_client_id = "sh-792ab2c7-0d03-4a90-b6d4-416779d61820"
        self.config.sh_client_secret = "zs2HqVN1UCRNx3P9dZhIlLBDTxOCQ4V4"
        self.config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
        self.config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
        self.catalog = SentinelHubCatalog(config=self.config)
        self.download_client = SentinelHubDownloadClient(config=self.config)
        self.query_results = None

    def request_products(
        self, bbox, time_interval=None, collection=DataCollection.SENTINEL2_L2A, max_cloud_coverage=20, limit=100
    ):
        """Query Sentinel Hub catalog for products.

        Args:
            bbox (list or BBox): Bounding box coordinates [min_x, min_y, max_x, max_y] or BBox object
            time_interval (tuple): Time range (start_date, end_date) as datetime objects or strings ('YYYY-MM-DD')
            collection (DataCollection): Sentinel data collection type
            max_cloud_coverage (float): Maximum cloud coverage percentage (0-100)
            limit (int): Maximum number of results to return

        Returns:
            dict: Query results
        """
        if time_interval is None:
            # Default to last 30 days if no time interval specified
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            time_interval = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        # Convert list to BBox if needed
        if isinstance(bbox, list):
            bbox = BBox(bbox, crs=CRS.WGS84)

        search_params = {
            "bbox": bbox,
            "time": time_interval,
            "collection": collection,
            "filter-lang": "cql2-text",
            "filter": "eo:cloud_cover < 5",
            "limit": limit,
        }

        self.query_results = self.catalog.search(**search_params)
        return self.query_results

    def filter_products(self, sort_by=None, ascending=False):
        """Convert results to DataFrame and filter/sort.

        Args:
            sort_by (str or list): Column(s) to sort by
            ascending (bool): Sort direction

        Returns:
            pandas.DataFrame: Filtered and sorted products
        """
        if self.query_results is None:
            raise ValueError("No query results. Call request_products first.")

        # Convert results to a DataFrame
        products = []
        for feature in self.query_results["features"]:
            product = {
                "id": feature["id"],
                "datetime": feature["properties"]["datetime"],
                "cloud_cover": feature["properties"].get("eo:cloud_cover"),
                "title": feature["properties"].get("title") or feature["id"],
            }

            # Add additional properties
            for key, value in feature["properties"].items():
                if key not in product:
                    product[key] = value

            products.append(product)

        products_df = pd.DataFrame(products)

        # Sort the DataFrame
        if sort_by is None:
            sort_by = ["datetime"]
        elif isinstance(sort_by, str):
            sort_by = [sort_by]

        return products_df.sort_values(sort_by, ascending=ascending)

    def download(self, product_id, path, bands=None, resolution=10):
        """Download a single product.

        Args:
            product_id (str): Product ID to download
            path (str): Directory to save downloaded files
            bands (list): List of bands to download (None for all available bands)
            resolution (int): Spatial resolution in meters

        Returns:
            str: Path to downloaded file
        """
        if not os.path.exists(path):
            os.makedirs(path)

        # Get product info
        product_info = self.catalog_api.get_product(product_id)

        # Prepare download request
        request = DownloadRequest(
            url=product_info["assets"]["data"]["href"],
            data_folder=path,
            filename=f"{product_id}.zip",
            data_collection=DataCollection.SENTINEL2_L2A,
            bands=bands,
        )

        # Download the product
        self.download_client.download([request])
        return os.path.join(path, f"{product_id}.zip")

    def download_all(self, product_df, path, bands=None, resolution=10):
        """Download all products in the DataFrame.

        Args:
            product_df (pandas.DataFrame): DataFrame of products to download
            path (str): Directory to save downloaded files
            bands (list): List of bands to download (None for all available bands)
            resolution (int): Spatial resolution in meters

        Returns:
            list: Paths to downloaded files
        """
        if not os.path.exists(path):
            os.makedirs(path)

        downloaded_paths = []
        for idx, row in product_df.iterrows():
            product_id = row["id"]
            file_path = self.download(product_id, path, bands, resolution)
            downloaded_paths.append(file_path)

        return downloaded_paths

    def get_product_info(self, product_id):
        """Get detailed information about a product.

        Args:
            product_id (str): Product ID

        Returns:
            dict: Product metadata
        """
        return self.catalog_api.get_product(product_id)
