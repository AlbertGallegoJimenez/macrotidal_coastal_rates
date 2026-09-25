import numpy as np
import pandas as pd
from scipy.stats import linregress, theilslopes

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
    Combines a time-based rolling median filter for shoreline stabilization
    with robust statistical regression (OLS and Theil-Sen) to calculate 
    evolution rates (m/year) for coastal profiles.
    """
    
    def __init__(
        self, 
        window='180D', 
        min_periods=1, 
        profile_col="profile_id", 
        date_col="datetime", 
        pos_col="shoreline_position_m"
    ):
        """
        Initializes the shoreline evolution analyzer.
        
        Parameters
        ----------
        window : str
            Time offset for the rolling window (e.g., '180D' for 180 days).
        min_periods : int
            Minimum number of observations in window required to compute a median.
        profile_col : str
            Column name for the unique profile/transect identifier.
        date_col : str
            Column name for the datetime variable.
        pos_col : str
            Column name for the cross-shore position to be smoothed and analyzed.
        """
        self.window = window
        self.min_periods = min_periods
        self.profile_col = profile_col
        self.date_col = date_col
        self.pos_col = pos_col
        self.rates_ = None  # Will store the resulting rates DataFrame

    def _smooth_data(self, df):
        """
        Applies the time-based rolling median grouped by profile safely,
        bypassing Pandas index alignment issues for duplicate timestamps.
        """
        df_smoothed = df.copy()
        
        # 1. Strict ordering to guarantee matrix alignment
        df_smoothed = df_smoothed.sort_values(by=[self.profile_col, self.date_col]).reset_index(drop=True)
        
        # 2. Temporarily set date as index for the time-based rolling operation
        df_temp = df_smoothed.set_index(self.date_col)
        
        # 3. Apply rolling median
        rolling_median = (
            df_temp.groupby(self.profile_col, sort=False)[self.pos_col]
            .rolling(window=self.window, min_periods=self.min_periods)
            .median()
        )
        
        # 4. Extract pure array to bypass Pandas index alignment
        smoothed_col_name = f"{self.pos_col}_smoothed"
        df_smoothed[smoothed_col_name] = rolling_median.values
        
        # 5. Drop initial NaNs generated by min_periods
        df_smoothed = df_smoothed.dropna(subset=[smoothed_col_name]).reset_index(drop=True)
        
        return df_smoothed

    def _calculate_rates(self, df_smoothed):
        """
        Calculates evolution rates (m/year) using both OLS and Theil-Sen estimators
        on the smoothed time series for each profile.
        """
        smoothed_col_name = f"{self.pos_col}_smoothed"
        rates_list = []
        
        for profile_id, group in df_smoothed.groupby(self.profile_col):
            # A minimum of 3 points is required to calculate variance and regression
            if len(group) < 3:
                continue 
                
            # Convert dates to continuous decimal years for accurate m/year metric
            days_from_epoch = (group[self.date_col] - pd.Timestamp("1970-01-01")).dt.days
            x_years = days_from_epoch / 365.2425
            y_pos = group[smoothed_col_name]
            
            # --- 1. Ordinary Least Squares (OLS) ---
            ols_res = linregress(x_years, y_pos)
            
            # --- 2. Theil-Sen Robust Regression ---
            # Returns: slope, intercept, lower CI bound, upper CI bound
            ts_res = theilslopes(y_pos, x_years, alpha=0.95)
            
            # Determine if the trend is statistically significant (p-value < 0.05)
            # and if the Theil-Sen 95% Confidence Interval doesn't cross zero.
            is_significant = (ols_res.pvalue < 0.05) and (np.sign(ts_res[2]) == np.sign(ts_res[3]))
            
            rates_list.append({
                self.profile_col: profile_id,
                "n_observations": len(group),
                "rate_ols_m_yr": round(ols_res.slope, 3),
                "r_squared": round(ols_res.rvalue**2, 3),
                "p_value": round(ols_res.pvalue, 4),
                "rate_theilsen_m_yr": round(ts_res[0], 3),
                "theilsen_ci_lower": round(ts_res[2], 3),
                "theilsen_ci_upper": round(ts_res[3], 3),
                "is_significant": is_significant
            })
            
        return pd.DataFrame(rates_list)

    def fit_transform(self, df):
        """
        Executes the full pipeline: smoothes the data and calculates the rates.
        
        Returns
        -------
        df_smoothed : pandas.DataFrame
            The dataset with the stabilized cross-shore positions.
            
        Note: The calculated rates are stored in the `rates_` attribute.
        """
        # Step 1: Smooth the noise
        df_smoothed = self._smooth_data(df)
        
        # Step 2: Calculate trends on the stabilized data
        self.rates_ = self._calculate_rates(df_smoothed)
        
        return df_smoothed