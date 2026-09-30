import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes
import statsmodels.api as sm

class SmartTidalFilter:
    """
    Evaluates and selects the optimal tidal window for shoreline extraction data
    based on data volume, seasonal concentration, data gaps, and tidal height.
    """
    
    def __init__(
        self, 
        min_years_span=5.0,
        max_gap_years=5.0, 
        lambda_1=1.0, 
        lambda_2=1.5,
        window_sizes=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
        step_size=0.1,
        beach_col="Playa",
        tide_col="NivelTotal",
        date_col="datetime"
    ):
        """
        Initializes the SmartTidalFilter with optimization hyperparameters and column names.
        
        Parameters
        ----------
        min_years_span : float
            Minimum required span of the data in years.
        max_gap_years : float
            Maximum allowed consecutive years without data.
        lambda_1 : float
            Penalty multiplier for wide tidal windows (reduces topographic noise).
        lambda_2 : float
            Penalty multiplier for obsolescence (distance to the most recent global data).
        window_sizes : list of float
            Tidal window widths to evaluate (in meters).
        step_size : float
            Tidal step for sliding the window (in meters).
        beach_col : str
            Column name for the beach or spatial grouping variable.
        tide_col : str
            Column name for the sea level / tide variable.
        date_col : str
            Column name for the date variable (must already be a pandas datetime object).
        """
        self.min_years_span = min_years_span
        self.max_gap_years = max_gap_years
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.window_sizes = window_sizes
        self.step_size = step_size
        self.beach_col = beach_col
        self.tide_col = tide_col
        self.date_col = date_col
        self.optimization_results = {}

    def _calculate_seasonal_concentration(self, months):
        """
        Calculates the circular concentration of months (Mean Resultant Length).
        Returns a value between 0 (completely dispersed) and 1 (all in the same month).
        """
        if len(months) == 0:
            return 0.0
            
        # Convert month (1-12) to an angle in radians (0 to 2*pi)
        angles = (months - 1) * (2 * np.pi / 12)
        
        sin_mean = np.mean(np.sin(angles))
        cos_mean = np.mean(np.cos(angles))
        
        # Mean Resultant Length (R)
        r_value = np.sqrt(sin_mean**2 + cos_mean**2)
        return r_value

    def _evaluate_window(self, df_window, z_min, z_max, beach_max_year):
        if df_window.empty or len(df_window) < 3:
            return -9999.0
            
        unique_dates = pd.Series(df_window[self.date_col].dt.date.unique()).sort_values()
        
        if len(unique_dates) < 3:
            return -9999.0

        # Check maximum temporal gap (Hard constraint)
        max_gap_days = pd.to_datetime(unique_dates).diff().dt.days.max()
        if max_gap_days / 365.25 > self.max_gap_years:
            return -9999.0
            
        # Check absolute temporal span (Hard constraint)
        window_min_year = unique_dates.min().year
        window_max_year = unique_dates.max().year
        window_span = window_max_year - window_min_year
        
        if window_span < self.min_years_span:
            return -9999.0

        # N: Number of unique temporal events
        n_events = len(unique_dates)
        
        # W: Tide Weight
        z_mean = df_window[self.tide_col].mean()
        w_tide = (z_mean - z_min) / (z_max - z_min) if z_max > z_min else 1.0
        
        # C: Seasonal Concentration
        months = pd.to_datetime(unique_dates).dt.month
        c_seasonal = self._calculate_seasonal_concentration(months)
        
        # Delta Z: Amplitude of the tidal window
        delta_z = df_window[self.tide_col].max() - df_window[self.tide_col].min()
        
        # T_gap: Obsolescence (distance to the beach's most recent data, not global)
        t_gap = beach_max_year - window_max_year
        
        # Fitness Equation (S)
        s_score = (np.log(n_events) * w_tide * c_seasonal) - (self.lambda_1 * delta_z) - (self.lambda_2 * t_gap)
        
        return s_score

    def fit_transform(self, df):
        """
        Applies the sliding window algorithm to group the dataframe by beach
        and extract the optimal subset of data.
        
        Parameters
        ----------
        df : pandas.DataFrame or geopandas.GeoDataFrame
            Must contain the columns specified in beach_col, tide_col, and date_col.
            The date_col must already be a pandas datetime object.
            
        Returns
        -------
        filtered_df : pandas.DataFrame or geopandas.GeoDataFrame
            The dataset containing only the intersections that fall within 
            the optimal tidal window for each beach.
        """
        df = df.copy()
        
        # Drop rows with NaNs in the essential columns
        df = df.dropna(subset=[self.date_col, self.tide_col, self.beach_col])
        
        global_min_year = df[self.date_col].dt.year.min()
        global_max_year = df[self.date_col].dt.year.max()
        
        filtered_subsets = []
        
        for beach_name, beach_df in df.groupby(self.beach_col):
            z_min = beach_df[self.tide_col].min()
            z_max = beach_df[self.tide_col].max()
            beach_max_year = beach_df[self.date_col].dt.year.max()
            
            best_score = -9999.0
            best_window = None
            
            # Sliding window generator
            for window_size in self.window_sizes:
                current_bottom = z_min
                while current_bottom + window_size <= z_max:
                    current_top = current_bottom + window_size
                    
                    window_df = beach_df[
                        (beach_df[self.tide_col] >= current_bottom) & 
                        (beach_df[self.tide_col] <= current_top)
                    ]
                    
                    score = self._evaluate_window(
                        window_df, z_min, z_max, beach_max_year
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_window = (current_bottom, current_top)
                        
                    current_bottom += self.step_size
            
            # Store optimization metadata for debugging or analysis
            self.optimization_results[beach_name] = {
                "best_score": best_score,
                "best_window": best_window
            }
            
            # If a valid window was found, extract the data
            if best_window is not None:
                optimal_df = beach_df[
                    (beach_df[self.tide_col] >= best_window[0]) & 
                    (beach_df[self.tide_col] <= best_window[1])
                ]
                filtered_subsets.append(optimal_df)
            else:
                print(f"Warning: No valid tidal window found for beach '{beach_name}'. "
                      "It did not pass the hard constraints (Coverage or Gaps).")
                
        if not filtered_subsets:
            return pd.DataFrame(columns=df.columns)
            
        return pd.concat(filtered_subsets, ignore_index=True)


class ShorelineEvolutionAnalyzer:
    """
    Stabilizes shoreline time series and calculates evolution rates (m/year) 
    using 5 different regression models: OLS, WLS (Sensor), WLS (Tide), 
    WLS (Combined), and Theil-Sen.
    """
    
    def __init__(
        self, 
        window='180D', 
        min_periods=1, 
        profile_col="profile_id", 
        date_col="datetime", 
        pos_col="shoreline_position_m",
        sensor_weight_col="sensor_weight", 
        tide_col="NivelTotal"
    ):
        """
        Initializes the shoreline evolution analyzer.
        
        Parameters
        ----------
        window : str
            Time offset for the rolling window (e.g., '180D').
        min_periods : int
            Minimum observations required in the window.
        profile_col : str
            Column name for the profile/transect identifier.
        date_col : str
            Column name for the datetime variable.
        pos_col : str
            Column name for the cross-shore position.
        sensor_weight_col : str
            Column containing pre-calculated sensor weights (e.g., S2=1.0, L5=0.33).
        tide_col : str
            Column containing tide elevation values to be used for weighting.
        """
        self.window = window
        self.min_periods = min_periods
        self.profile_col = profile_col
        self.date_col = date_col
        self.pos_col = pos_col
        self.sensor_weight_col = sensor_weight_col
        self.tide_col = tide_col
        self.rates_ = None

    def _smooth_data(self, df):
        """
        Applies rolling median to positions and rolling mean to weights/tides.
        """
        df_smoothed = df.copy()
        df_smoothed = df_smoothed.sort_values(by=[self.profile_col, self.date_col]).reset_index(drop=True)
        df_temp = df_smoothed.set_index(self.date_col)
        
        # 1. Smooth Shoreline Position (Median)
        rolling_pos = (
            df_temp.groupby(self.profile_col, sort=False)[self.pos_col]
            .rolling(window=self.window, min_periods=self.min_periods)
            .median()
        )
        df_smoothed[f"{self.pos_col}_smoothed"] = rolling_pos.values
        
        # 2. Smooth Sensor Weight (Mean)
        if self.sensor_weight_col in df_smoothed.columns:
            rolling_sensor = (
                df_temp.groupby(self.profile_col, sort=False)[self.sensor_weight_col]
                .rolling(window=self.window, min_periods=self.min_periods)
                .mean()
            )
            df_smoothed[f"{self.sensor_weight_col}_smoothed"] = rolling_sensor.values
            
        # 3. Smooth Tide Level (Mean)
        if self.tide_col in df_smoothed.columns:
            rolling_tide = (
                df_temp.groupby(self.profile_col, sort=False)[self.tide_col]
                .rolling(window=self.window, min_periods=self.min_periods)
                .mean()
            )
            df_smoothed[f"{self.tide_col}_smoothed"] = rolling_tide.values
            
        # Drop NaNs generated by min_periods
        df_smoothed = df_smoothed.dropna(subset=[f"{self.pos_col}_smoothed"]).reset_index(drop=True)
        return df_smoothed

    def _fit_linear_model(self, x, y, weights=None):
        """
        Helper method to fit OLS or WLS using statsmodels.
        """
        X_sm = sm.add_constant(x)
        if weights is None:
            model = sm.OLS(y, X_sm).fit()
        else:
            model = sm.WLS(y, X_sm, weights=weights).fit()
            
        slope = model.params[1] if len(model.params) > 1 else 0.0
        pvalue = model.pvalues[1] if len(model.pvalues) > 1 else 1.0
        return slope, model.rsquared, pvalue

    def _calculate_rates(self, df_smoothed):
        """
        Calculates evolution rates using 5 models: 
        OLS, WLS(Sensor), WLS(Tide), WLS(Combined), and Theil-Sen.
        """
        pos_sm = f"{self.pos_col}_smoothed"
        sens_sm = f"{self.sensor_weight_col}_smoothed"
        tide_sm = f"{self.tide_col}_smoothed"
        
        rates_list = []
        
        for profile_id, group in df_smoothed.groupby(self.profile_col):
            if len(group) < 3:
                continue 
                
            days_from_epoch = (group[self.date_col] - pd.Timestamp("1970-01-01")).dt.days
            x_years = days_from_epoch.values / 365.2425
            y_pos = group[pos_sm].values
            
            # --- Prepare Weights ---
            # 1. Sensor Weights
            w_sensor = group[sens_sm].values if sens_sm in group.columns else np.ones_like(y_pos)
            
            # 2. Tide Weights (Scaled Min-Max to [0.1, 1.0] to prevent zero-weight WLS failure)
            w_tide = np.ones_like(y_pos)
            if tide_sm in group.columns:
                tide_vals = group[tide_sm].values
                t_min, t_max = tide_vals.min(), tide_vals.max()
                if t_max > t_min:
                    w_tide = 0.1 + 0.9 * ((tide_vals - t_min) / (t_max - t_min))
            
            # 3. Combined Weights (Even distribution)
            w_combined = (w_sensor + w_tide) / 2.0
            
            # --- Calculate Models ---
            # 1. OLS
            ols_slope, ols_r2, ols_pval = self._fit_linear_model(x_years, y_pos)
            
            # 2. WLS (Sensor Only)
            wls_sens_slope, wls_sens_r2, wls_sens_pval = self._fit_linear_model(x_years, y_pos, w_sensor)
            
            # 3. WLS (Tide Only)
            wls_tide_slope, wls_tide_r2, wls_tide_pval = self._fit_linear_model(x_years, y_pos, w_tide)
            
            # 4. WLS (Combined)
            wls_comb_slope, wls_comb_r2, wls_comb_pval = self._fit_linear_model(x_years, y_pos, w_combined)
            
            # 5. Theil-Sen Robust
            ts_res = theilslopes(y_pos, x_years, alpha=0.95)
            ts_slope, ts_intercept, ts_ci_low, ts_ci_up = ts_res
            
            # Significance (Based on Theil-Sen confidence interval stability)
            is_significant = (ols_pval < 0.05) and (np.sign(ts_ci_low) == np.sign(ts_ci_up))
            
            rates_list.append({
                self.profile_col: profile_id,
                "n_observations": len(group),
                
                # OLS Metrics
                "rate_ols": round(ols_slope, 3),
                "r2_ols": round(ols_r2, 3),
                
                # WLS Sensor Metrics
                "rate_wls_sensor": round(wls_sens_slope, 3),
                "r2_wls_sensor": round(wls_sens_r2, 3),
                
                # WLS Tide Metrics
                "rate_wls_tide": round(wls_tide_slope, 3),
                "r2_wls_tide": round(wls_tide_r2, 3),
                
                # WLS Combined Metrics
                "rate_wls_combined": round(wls_comb_slope, 3),
                "r2_wls_combined": round(wls_comb_r2, 3),
                
                # Theil-Sen Metrics
                "rate_theilsen": round(ts_slope, 3),
                "ts_ci_lower": round(ts_ci_low, 3),
                "ts_ci_upper": round(ts_ci_up, 3),
                
                # Global Significance
                "is_significant": is_significant
            })
            
        return pd.DataFrame(rates_list)

    def fit_transform(self, df):
        """
        Executes the full pipeline: smoothes the data and calculates the rates.
        """
        df_smoothed = self._smooth_data(df)
        self.rates_ = self._calculate_rates(df_smoothed)
        return df_smoothed