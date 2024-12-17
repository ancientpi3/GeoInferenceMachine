class GeoInferenceMachine:
    def __init__(self, geotiff_path,
                 tile_dimensions,
                 tiles_folder_path,
                 preds_folder_path,
                 model_path=None,
                 ):
        self.geotiff_path = geotiff_path
        self.tile_dimensions = tile_dimensions
        self.tiles_folder_path = tiles_folder_path
        self.tile_input_data_geo = {} # New dictionary to store geo-coordinates
        self.preds_folder_path = preds_folder_path
        self.model_path = model_path
        self.model = None
        if not os.path.exists(self.tiles_folder_path):
            os.makedirs(self.tiles_folder_path)
        if not os.path.exists(self.preds_folder_path):
            os.makedirs(self.preds_folder_path)
        # Add other instance variables as needed

    def multiband_to_png(self, tile):
        """
        Converts a multi-band image to a PNG image.
        Override this function to implement your custom conversion logic.
        """
        # Assuming the tile is a numpy array with shape (bands, height, width)
        # Convert each band to an 8-bit grayscale image
        images = []
        for band in range(tile.shape[0]):
            band_data = tile[band,:, :]

            # Normalize to 0-255 range
            min_val = np.min(band_data)
            max_val = np.max(band_data)
            normalized_band = ((band_data - min_val) / (max_val - min_val)) * 255

            # Convert to 8-bit unsigned integer
            normalized_band = normalized_band.astype(np.uint8)
            images.append(Image.fromarray(normalized_band))

    def tile_input_data(self):
        with rasterio.open(self.geotiff_path) as src:
            width = src.width
            height = src.height
            tile_width, tile_height = self.tile_dimensions
            tiles = []
            total_tiles = (height // tile_height) * (width // tile_width)
            with tqdm(total=total_tiles, desc="Tiling Progress") as pbar:
              for i in range(0, height, tile_height):
                  for j in range(0, width, tile_width):
                      window = rasterio.windows.Window(j, i, tile_width, tile_height)
                      tile = src.read(window=window)
                      if tile.shape[1] != tile_height or tile.shape[2] != tile_width:
                          tile = src.read(window=window,
                                          boundless=True,
                                          # fill_value=0,  # Value for out-of-bounds areas
                                          # out_shape=(height, width)
                                          )


                      png_image = self.multiband_to_png(tile)
                      tile_filename = os.path.join(self.tiles_folder_path, f"tile_{i}_{j}.png")
                      png_image.save(tile_filename)

                      # Store geo-coordinates
                      transform = src.window_transform(window)
                      self.tile_input_data_geo[tile_filename] = {
                          'transform': transform.to_gdal(),
                          'crs': src.crs.to_string()
                      }

                      pbar.update(1)

    def png_pred_to_geotiff(self, png_path, geo_info_path=None, pred=None):
            if geo_info_path is None:
                geo_info_path = self.tile_input_data_geo
            geo_info = geo_info_path[png_path]
            transform_tuple = geo_info['transform']
            transform = rasterio.Affine.from_gdal(*transform_tuple)
            crs = geo_info['crs']
            img_array = np.array(pred)
            # img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
            # Construct the output path within self.preds_folder_path
            output_path = os.path.join(self.preds_folder_path, os.path.basename(png_path).replace('.png', '_pred.tif'))
            with rasterio.open(output_path, 'w', driver='GTiff',
                                height=img_array.shape[0], width=img_array.shape[1],
                                count=img_array.shape[2] if len(img_array.shape) > 2 else 1,
                                dtype=rasterio.float32,
                                crs=crs,
                                transform=transform) as dst:
                if len(img_array.shape) > 2:
                    dst.write(img_array)
                else:
                    dst.write(img_array, 1)
    
    def combine_geotiffs(self, save_location, search_location):
      """
      Combines all GeoTIFFs in self.preds_folder_path into a single GeoTIFF.

      Args:
          save_location (str): The path to save the combined GeoTIFF.
      """
      if not search_location:
          search_location = self.preds_folder_path
      search_criteria = os.path.join(search_location, '**/*.tif')
      geotiff_files = glob.glob(search_criteria)

      if not geotiff_files:
          print(f"No GeoTIFF files found in {search_location}")
          return

      src_files_to_mosaic = []
      for fp in geotiff_files:
          src = rasterio.open(fp)
          src_files_to_mosaic.append(src)

      mosaic, out_trans = merge(src_files_to_mosaic)

      # Copy the metadata
      out_meta = src.meta.copy()

      # Update the metadata
      out_meta.update({"driver": "GTiff",
                      "height": mosaic.shape[1],
                      "width": mosaic.shape[2],
                      "transform": out_trans,
                      "crs": src.crs})

      # Write the mosaic raster to disk
      with rasterio.open(save_location, "w", **out_meta) as dest:
          dest.write(mosaic)

      # Close the datasets
      for src in src_files_to_mosaic:
          src.close()


    def initialize_model(self):
        """
        Initializes the machine learning model.
        Override this function to implement your custom model initialization logic.
        """
        raise NotImplementedError("Subclasses must implement this method")


    def generate_inferences(self):
        """
        Runs ML inference on a tile.
        Override this function to implement your custom inference logic.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def run(self):
        """
        Orchestrates the tiling, preprocessing, and inference process.
        """
        self.tile_input_data()
        # ... (rest of your run method)
