import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import os

def write_geotiff(mltps_in, out_names=None, overwrite=True, output_dir=None):
    if output_dir is None:
        output_dir = os.getcwd()
    
    n_spln = mltps_in[0]['n_layers']
    
    results_summary = []
    
    for i in range(n_spln):
        layer_data = mltps_in[i]
        final_raster = layer_data['final'] # This should be a numpy array or rasterio object
        # In our implementation, let's assume it's a dictionary with 'data', 'transform', 'crs'
        
        layer_name = layer_data['summary'].iloc[0, 0]
        if out_names is not None:
            layer_name = out_names[i]
            
        output_filename = os.path.join(output_dir, f"{layer_name}.tif")
        
        # Assume final_raster is a dict with meta and data
        meta = final_raster['meta']
        data = final_raster['data']
        
        with rasterio.open(output_filename, 'w', **meta) as dst:
            dst.write(data, 1)
        
        print(f"Layer {layer_name} saved to: {output_filename}")
        
        results_summary.append(layer_data['summary'])
    
    # Write summary CSV
    summary_table = pd.concat(results_summary, ignore_index=True)
    import random
    rand_id = random.randint(100000, 999999)
    csv_filename = f"MACHISPLIN_results_{rand_id}.csv"
    summary_table.to_csv(csv_filename, index=False)
    
    legend = [
        "",
        "R2 Final: ensemble of the best models & thin-plate-spline of the residuals of the ensemble model",
        "Best model legend: The quantity of letters depicts the number of models ensembled.",
        "The letters themselves depict the model algorithm: b = boosted regression trees (BRT);",
        "g = generalized additive model (GAM); m = multivariate adaptive regression splines (MARS);",
        "v = support vector machines (SVM); r = random forests (RF); n = neural networks (NN)",
        "The ensemble weights is percentage that each algorithm contributed to the ensemble model",
        "NOTE: if 'R2 Ensemble' is greater than 'R2 Final', then the output model is only the ensembled model (the thin-plate-spline of residuals were not used)"
    ]
    with open(csv_filename, 'a') as f:
        f.write("\n".join(legend))

def write_loadings(mltps_in):
    n_spln = mltps_in[0]['n_layers']
    for i in range(n_spln):
        layer_name = mltps_in[i]['summary'].iloc[0, 0]
        filename = f"{layer_name}_model_loadings.txt"
        with open(filename, 'w') as f:
            f.write(str(mltps_in[i]['var_imp']))
            
def write_residuals(mltps_in):
    n_spln = mltps_in[0]['n_layers']
    for i in range(n_spln):
        layer_name = mltps_in[i]['summary'].iloc[0, 0]
        filename = f"{layer_name}_residuals.csv"
        residuals_df = pd.DataFrame(mltps_in[i]['residuals'], columns=['residuals', 'long', 'lat'])
        residuals_df.to_csv(filename, index=False)
