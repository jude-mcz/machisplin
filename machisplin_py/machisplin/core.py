import numpy as np
import pandas as pd
import rasterio
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from .utils import kfold, calc_deviance, roc_score, calibration
from .models import MACHISPLINModel
from .tiling import tiles_create, tiles_merge
import os

def mltps(int_values, covar_ras_path, tps=True, smooth_outputs_only=False, trouble=False):
    """
    Machine Learning Ensemble & Thin-Plate-Spline Interpolation.
    """
    print("IMPORTANT: all data input are assumed to be WGS1984 projection/datum")
    
    with rasterio.open(covar_ras_path) as src:
        ras_meta = src.meta
        ras_bounds = src.bounds
        n_covars = src.count
        covar_names = [src.descriptions[i] or f"covar_{i+1}" for i in range(n_covars)]
        # Add LONG and LAT to names
        covar_names += ["LONG", "LAT"]
        ras_data = src.read()
        
    # Get LONG and LAT for each cell
    with rasterio.open(covar_ras_path) as src:
        rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing='ij')
        lons, lats = rasterio.transform.xy(src.transform, rows, cols)
        lons = np.array(lons)
        lats = np.array(lats)
        
    # Stack rasters with LONG and LAT
    ras_data_full = np.concatenate([ras_data, lons[np.newaxis, :, :], lats[np.newaxis, :, :]], axis=0)
    n_covars_full = n_covars + 2
    
    # Preparation
    # int_values: long, lat, resp1, resp2, ...
    i_lyrs = int_values.shape[1]
    n_spln = i_lyrs - 2
    
    # Extract values from rasters at point locations
    # We already have coords from int_values. We need to sample our new ras_data_full
    # Instead of rasterio.sample, since we have the data in memory now
    def sample_ras_full(longs, lats, ras_data, transform):
        # transform from coord to pixel
        import rasterio.transform
        inv_trans = ~transform
        rows, cols = rasterio.transform.rowcol(transform, longs, lats)
        # Ensure rows/cols are within bounds
        rows = np.clip(rows, 0, ras_data.shape[1] - 1)
        cols = np.clip(cols, 0, ras_data.shape[2] - 1)
        return ras_data[:, rows, cols].T

    ras_val = sample_ras_full(int_values['long'].values, int_values['lat'].values, ras_data_full, src.transform)
    
    # Merge sampled data
    my_data = pd.concat([int_values.iloc[:, :i_lyrs], pd.DataFrame(ras_val, columns=covar_names)], axis=1)
    my_data_full_count = len(my_data)
    my_data = my_data.dropna()
    
    if len(my_data) / my_data_full_count < 0.75:
        print(f"Warning! {my_data_full_count - len(my_data)} points fell outside rasters.")
    if len(my_data) < 40:
        print(f"Warning! {len(my_data)} sites are too few for this method (min 40).")
        
    # For each climate variable (resp column)
    omega = []
    for i in range(n_spln):
        resp_col = int_values.columns[i+2]
        print(f"Starting downscaling for {resp_col}")
        
        # Prepare data for current variable
        X = my_data[covar_names].values
        y = my_data[resp_col].values
        
        # Step 1: K-fold CV and Ensemble Weighting
        nfolds = 10
        if len(my_data) > 4000:
            nfolds = 10
        elif len(my_data) > 60:
            nfolds = 10
        elif len(my_data) > 39:
            nfolds = 2
        
        kf = kfold(len(my_data), k=nfolds)
        
        # Models to evaluate
        if smooth_outputs_only:
            model_types = ['GAM', 'NN', 'MARS', 'SVM']
        else:
            model_types = ['BRT', 'GAM', 'NN', 'MARS', 'RF', 'SVM']
            
        fold_residuals = {mt: [] for mt in model_types}
        
        for fold in range(1, nfolds + 1):
            print(f"Fold {fold}/{nfolds}")
            train_idx = kf != fold
            test_idx = kf == fold
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train each model and get residuals on test set
            for mt in model_types:
                model = MACHISPLINModel(mt)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                fold_residuals[mt].extend(y_test - preds)
                
        # Optimization to find best weights
        def objective(weights, residuals):
            # weights sum to 1? R code uses k1/(k1+k2+...)
            weights = np.array(weights)
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                return 1e10
            
            weighted_res = np.zeros(len(residuals[model_types[0]]))
            for i, mt in enumerate(model_types):
                weighted_res += (np.array(residuals[mt]) * weights[i]) / weights_sum
            return np.sum(weighted_res**2)
            
        init_weights = [0.5] * len(model_types)
        res = minimize(objective, init_weights, args=(fold_residuals,), bounds=[(0, 1)] * len(model_types))
        best_weights = res.x
        best_weights /= np.sum(best_weights)
        
        # Step 2: Final Ensemble Model
        print(f"Fitting final models with weights: {dict(zip(model_types, best_weights))}")
        final_models = {}
        for i, mt in enumerate(model_types):
            if best_weights[i] > 0.05:
                model = MACHISPLINModel(mt)
                model.fit(X, y)
                final_models[mt] = (model, best_weights[i])
        
        # Compute final ensemble prediction on rasters
        # For memory efficiency, we might need to do this in chunks
        # But for now, let's assume ras_data fits in memory
        X_ras = ras_data_full.reshape(n_covars_full, -1).T
        # X_ras should be (n_cells, n_covars_full)
        ensemble_pred = np.zeros(X_ras.shape[0])
        for mt, (model, weight) in final_models.items():
            ensemble_pred += model.predict(X_ras) * weight
        ensemble_pred /= sum(w for m, w in final_models.values())
        
        # Residuals on full training set
        train_ensemble_pred = np.zeros(len(y))
        for mt, (model, weight) in final_models.items():
            train_ensemble_pred += model.predict(X) * weight
        train_ensemble_pred /= sum(w for m, w in final_models.values())
        ensemble_residuals = y - train_ensemble_pred
        
        # Final R2 calculation
        rss_m = np.sum(ensemble_residuals**2)
        tss = np.sum((y - np.mean(y))**2)
        rsq_model = 1 - (rss_m / tss)
        
        # Step 3: TPS of Residuals
        if tps:
            print("Performing Thin Plate Spline of residuals...")
            # For TPS, we use longitude and latitude
            X_tps = my_data[['long', 'lat']].values
            # RBFInterpolator with thin_plate_spline
            tps_model = RBFInterpolator(X_tps, ensemble_residuals, kernel='thin_plate_spline')
            
            # Predict TPS on all cells (using their long/lat)
            # We need long/lat for each cell
            with rasterio.open(covar_ras_path) as src:
                rows, cols = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing='ij')
                lons, lats = rasterio.transform.xy(src.transform, rows, cols)
                lons = np.array(lons).flatten()
                lats = np.array(lats).flatten()
                
            X_ras_coords = np.column_stack((lons, lats))
            tps_residuals = tps_model(X_ras_coords)
            
            final_pred = ensemble_pred + tps_residuals
            
            # Recalculate residuals and R2 after TPS
            # We need to sample final_pred at training points
            # final_pred is (n_cells,)
            # We can use the same sample_ras_full logic but for a single layer
            def sample_single_layer(longs, lats, data, transform):
                rows, cols = rasterio.transform.rowcol(transform, longs, lats)
                rows = np.clip(rows, 0, data.shape[0] - 1)
                cols = np.clip(cols, 0, data.shape[1] - 1)
                return data[rows, cols]
            
            final_pred_ras = final_pred.reshape(src.height, src.width)
            f_actual = sample_single_layer(my_data['long'].values, my_data['lat'].values, final_pred_ras, src.transform)
            rss_final = np.sum((y - f_actual)**2)
            rsq_final = 1 - (rss_final / tss)
            
            # If rsq_final < rsq_model, we should ideally stick with the ensemble model
            if rsq_final < rsq_model:
                print("Warning: TPS did not improve R2. Using ensemble model only.")
                final_pred = ensemble_pred
                rsq_final = rsq_model
        else:
            final_pred = ensemble_pred
            rsq_final = rsq_model
            
        # Reshape final_pred back to raster shape
        with rasterio.open(covar_ras_path) as src:
            out_data = final_pred.reshape(src.height, src.width)
            out_meta = src.meta.copy()
            out_meta.update(count=1, dtype='float32')
            
        # Return object similar to R structure
        l = {
            'n_layers': n_spln,
            'final': {'data': out_data, 'meta': out_meta},
            'residuals': np.column_stack((ensemble_residuals, my_data['long'], my_data['lat'])),
            'summary': pd.DataFrame([[resp_col, ":".join(final_models.keys()), 
                                      ":".join([str(round(w*100,1)) for w in [f[1] for f in final_models.values()]]), 
                                      rsq_model, rsq_final]], 
                                    columns=["layer", "best model(s):", "ensemble weights:", "r2 ensemble:", "r2 final:"]),
            'var_imp': {mt: model[0].get_importance(covar_names) for mt, model in final_models.items()}
        }
        omega.append(l)
        
    return omega
