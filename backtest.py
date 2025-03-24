# -*- coding: utf-8 -*-
# noinspection SpellCheckingInspection
"""
|\---/|
| o_o |
_\_^_/----- Replicats.ai ----- 

    Sample backtests

Copyright (2025) - Replicats.ai
@author: jorge@replicats.ai (TL: @quantamentalguy)
"""
import pandas as pd
import numpy as np
from scipy import stats

def drawdowns(returns: pd.Series):
    """Calculate maximum drawdown and its duration"""
    cumulative_returns =  returns.apply(np.exp).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = cumulative_returns - rolling_max
    drawdowns_perc = cumulative_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Calculate drawdown duration
    end_idx = drawdowns_perc.idxmin()
    peak_idx = rolling_max.loc[:end_idx].idxmax()
    recovery_idx = drawdowns_perc.loc[end_idx:].gt(0).idxmax() if any(drawdowns_perc.loc[end_idx:] >= 0) else returns.index[-1]
    recovery_idx = pd.to_datetime(recovery_idx)
    peak_idx = pd.to_datetime(peak_idx)
    drawdown_duration = (recovery_idx - peak_idx).days
            
    return drawdowns, drawdowns_perc, max_drawdown, drawdown_duration


def calculate_ratios(returns: pd.Series, 
                     risk_free_rate: float = 0.0, 
                     periods_per_year: int = 365, 
                     rolling_window: int = None, 
                     min_periods: int = None) -> dict:
    """
    Calculate various risk-adjusted return metrics including Sharpe, Sortino, Calmar, 
    Treynor, and Information ratios.
    
    Parameters:
        returns (pd.Series): Series of periodic returns
        risk_free_rate (float): Annual risk-free rate (default: 0.0)
        periods_per_year (int): Number of periods in a year (default: 365 for daily data)
        rolling_window (int): Window size for rolling calculations (default: None)
        min_periods (int): Minimum periods for rolling calculations (default: None)
    
    Returns:
        dict: Dictionary containing various risk-adjusted return metrics
    """
    # removing issing values (flat)
    returns = returns.bfill(axis=0).ffill(axis=0)

    # Convert annual risk-free rate to periodic    
    rf_periodic = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_periodic
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate annualized return
    total_years = len(returns) / periods_per_year
    annualized_return = (cum_returns.iloc[-1] ** (1/total_years)) - 1
    
    # Calculate volatility
    annual_std = returns.std() * np.sqrt(periods_per_year)
    
    # Calculate downside volatility
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(periods_per_year)
    
    # Calculate max drawdown and duration
    _, _, max_dd, dd_duration = drawdowns(returns)
    
    # Calculate various ratios
    mean_excess_return = excess_returns.mean() * periods_per_year
    
    # Sharpe Ratio
    sharpe_ratio = mean_excess_return / annual_std if annual_std != 0 else np.nan
    
    # Sortino Ratio
    sortino_ratio = mean_excess_return / downside_std if downside_std != 0 else np.nan
    
    # Calmar Ratio
    calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else np.nan
    
    # Calculate Beta (using market returns as excess returns for simplicity)
    rolling_returns_std = returns.rolling(window=min(365, len(returns))).std()
    beta = rolling_returns_std / returns.std() if returns.std() != 0 else 1
    beta = np.float64(beta.dropna().mean())
    
    # Treynor Ratio 
    treynor_ratio = mean_excess_return / beta if beta != 0 else np.nan    
    
    # Information Ratio (using excess returns as active returns)
    information_ratio = mean_excess_return / excess_returns.std() if excess_returns.std() != 0 else np.nan
    
    # Omega Ratio
    threshold = 0  # Can be adjusted based on minimum acceptable return
    omega_ratio = len(returns[returns > threshold]) / len(returns[returns <= threshold])
    
    results = {
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'treynor_ratio': treynor_ratio,
        'information_ratio': information_ratio,
        'omega_ratio': omega_ratio,
        'annualized_return': annualized_return,
        'annualized_volatility': annual_std,
        'downside_volatility': downside_std,
        'max_drawdown': max_dd,
        'drawdown_duration': dd_duration,
        'beta': beta,
        'excess_returns': excess_returns
    }
    
    # Calculate rolling metrics if window is specified
    if rolling_window is not None:
        min_periods = min_periods or rolling_window
        
        # Rolling returns and volatility
        rolling_returns = returns.rolling(window=rolling_window, min_periods=min_periods)
        rolling_excess_returns = excess_returns.rolling(window=rolling_window, min_periods=min_periods)
        
        # Rolling Sharpe
        rolling_sharpe = (rolling_excess_returns.mean() * periods_per_year) / \
                        (rolling_returns.std() * np.sqrt(periods_per_year))
        
        # Rolling Sortino
        def rolling_sortino(x):
            excess_mean = (x - rf_periodic).mean() * periods_per_year
            downside = x[x < 0]
            downside_std = np.sqrt(np.mean(downside**2)) * np.sqrt(periods_per_year)
            return excess_mean / downside_std if downside_std != 0 else np.nan
        
        rolling_sortino = returns.rolling(window=rolling_window, min_periods=min_periods).apply(rolling_sortino)
        
        # Rolling Calmar
        def rolling_calmar(x):
            cum_rets = (1 + x).cumprod()
            max_dd = (cum_rets / cum_rets.expanding().max() - 1).min()
            ann_ret = (cum_rets.iloc[-1] ** (periods_per_year/len(x))) - 1
            return ann_ret / abs(max_dd) if max_dd != 0 else np.nan
        
        rolling_calmar = returns.rolling(window=rolling_window, min_periods=min_periods).apply(rolling_calmar)
        
        results.update({
            'rolling_sharpe_ratio': rolling_sharpe,
            'rolling_sortino_ratio': rolling_sortino,
            'rolling_calmar_ratio': rolling_calmar
        })
    
    return results

def calculate_var_metrics(returns: pd.Series, confidence_level: float = 0.95):
    """Calculate various Value at Risk (VaR) metrics"""
    # converts log-returns to daily returns convetion
    # Historical VaR c  
    hist_var = np.percentile(returns, (1 - confidence_level) * 100)
    
    # Parametric VaR (assuming normal distribution)
    mean = returns.mean()
    std = returns.std()
    param_var = stats.norm.ppf(1 - confidence_level, mean, std)
    
    # Conditional VaR (Expected Shortfall)
    cvar = returns[returns <= hist_var].mean()
    
    # Modified VaR using Cornish-Fisher expansion
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    z_score = stats.norm.ppf(confidence_level)
    modified_var = -(mean + std * (z_score + 
                                 (z_score**2 - 1) * skew / 6 +
                                 (z_score**3 - 3*z_score) * kurt / 24 -
                                 (2*z_score**3 - 5*z_score) * skew**2 / 36))
    
    # returns 1 day absolute values for a linear portifolio
    return {
        'historical_var': hist_var,
        'parametric_var': param_var,
        'conditional_var': cvar,
        'modified_var': modified_var
    }


def calculate_tracking_metrics(returns: pd.Series, benchmark_returns: pd.Series):
    """Calculate benchmark comparison metrics"""
    # Tracking Error
    tracking_diff = returns - benchmark_returns
    tracking_error = tracking_diff.std() * np.sqrt(365)
    
    # Information Ratio with benchmark
    active_return = returns.mean() - benchmark_returns.mean()
    information_ratio = (active_return * 365) / tracking_error if tracking_error != 0 else np.nan
    
    # Beta and Alpha
    covar = np.cov(returns, benchmark_returns)[0][1]
    benchmark_var = np.var(benchmark_returns)
    beta = covar / benchmark_var if benchmark_var != 0 else np.nan
    
    # Calculate alpha (annualized)
    alpha = (returns.mean() - beta * benchmark_returns.mean()) * 365
    
    # R-squared
    corr_matrix = np.corrcoef(returns, benchmark_returns)
    r_squared = corr_matrix[0][1]**2
    
    # Up/Down Capture Ratios
    up_markets = benchmark_returns > 0
    down_markets = benchmark_returns < 0
    
    up_capture = (returns[up_markets].mean() / benchmark_returns[up_markets].mean()) if len(benchmark_returns[up_markets]) > 0 else np.nan
    down_capture = (returns[down_markets].mean() / benchmark_returns[down_markets].mean()) if len(benchmark_returns[down_markets]) > 0 else np.nan
    
    return {
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'alpha': alpha,
        'beta': beta,
        'r_squared': r_squared,
        'up_capture': up_capture,
        'down_capture': down_capture,
        'tracking_diff': tracking_diff
    }

def calculate_ratios_risk(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 365, 
                    rolling_window: int = None, benchmark_returns: pd.Series = None,
                    var_confidence_level: float = 0.95) -> dict:
    """
    Calculate comprehensive risk metrics including VaR and benchmark comparison.
    
    Parameters:
        returns (pd.Series): Series of periodic returns
        risk_free_rate (float): Annual risk-free rate (default: 0.0)
        periods_per_year (int): Number of periods in a year (default: 365)
        rolling_window (int): Window size for rolling calculations (default: None)
        benchmark_returns (pd.Series): Benchmark returns series (default: None)
        var_confidence_level (float): Confidence level for VaR calculations (default: 0.95)
    
    Returns:
        dict: Dictionary containing all risk metrics
    """
    # Get base metrics from previous implementation
    base_results = calculate_ratios(returns, risk_free_rate, periods_per_year, rolling_window)
    
    # Calculate VaR metrics
    var_results = calculate_var_metrics(returns, var_confidence_level)
    
    # Combine results
    results = {**base_results, **var_results}
    
    # Add benchmark comparison if provided
    if benchmark_returns is not None:
        benchmark_metrics = calculate_tracking_metrics(returns, benchmark_returns)
        results.update(benchmark_metrics)
        
        # Calculate rolling metrics with benchmark
        if rolling_window is not None:
            rolling_tracking_error = (returns - benchmark_returns).rolling(window=rolling_window).std() * np.sqrt(periods_per_year)
            
            def rolling_alpha_beta(window_data):
                window_returns = window_data['returns']
                window_benchmark = window_data['benchmark']
                covar = np.cov(window_returns, window_benchmark)[0][1]
                benchmark_var = np.var(window_benchmark)
                beta = covar / benchmark_var if benchmark_var != 0 else np.nan
                alpha = (window_returns.mean() - beta * window_benchmark.mean()) * periods_per_year
                return pd.Series({'alpha': alpha, 'beta': beta})
            
            combined_data = pd.DataFrame({
                'returns': returns,
                'benchmark': benchmark_returns
            })
            
            rolling_ab = combined_data.rolling(window=rolling_window).apply(
                lambda x: rolling_alpha_beta(x)
            )
            
            results.update({
                'rolling_tracking_error': rolling_tracking_error,
                'rolling_alpha': rolling_ab['alpha'],
                'rolling_beta': rolling_ab['beta']
            })
    
    return results

def simulate_gbm(S0: float, mu: float, sigma: float, T: float, N: int, n_sims: int, 
                 random_seed=None):
    """
    Simulate Geometric Brownian Motion.
    
    Args:
        S0 (float): Initial price
        mu (float): Drift (annualized)
        sigma (float): Volatility (annualized)
        T (float): Time period in years
        N (int): Number of time steps
        num_sims (int): Number of simulation paths
        random_seed (int, optional): Random seed for reproducibility
        
    Returns:
        np.array: Simulated price paths of shape (num_sims, N+1)
    """    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    N = int(N)
    dt = float(T) / N
    # Initialize price array (n_simulations x N+1)
    S = np.zeros((n_sims, N + 1))
    S[:, 0] = S0
    
    # Standard GBM formula: S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)
    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_sims)
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return S


def monte_carlo_backtest(returns: pd.Series, px_last: float, 
                          n_simulations: int=1000, confidence_level: float=0.95, 
                          times = None, title=None):
    """
    Perform Monte Carlo backtesting (GBM) on a trading strategy

    REMARK: this is a SIMPLIFIED version! The correct approach is to
        simulate scenarios for all strategy risk factors and perfrom
        the backtest on full valuation
    
    Parameters:
        returns (array-like): Historical returns of the strategy
        n_simulations (int): Number of Monte Carlo simulations
        confidence_level (float): Confidence level for risk metrics
    
    Returns:
        dict: Various risk metrics and performance statistics
    """
    returns = np.array(returns)
    n = len(returns)
    s0 = px_last * np.exp(-np.sum(returns))
    
    # calibrate GBM (daily returns)
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    # simulate paths
    paths = simulate_gbm(s0, mu, sigma, float(n), int(n), int(n_simulations))
    
    # calculates expected value
    final_prices = paths[:, -1]
    expected_price = np.mean(final_prices)
    
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2
    upper_percentile = 1 - lower_percentile
    
    def max_drawdowns(returns):
        _, _, value, _ = drawdowns(pd.Series(returns))
        return value


    paths_returns = np.diff(paths, axis=1)
    var = np.percentile(paths_returns, (1 - confidence_level) * 100, axis=1)
    
    cvar = np.zeros(n_simulations)
    max_drawdown = np.zeros(n_simulations)

    for i in range(n_simulations):
        path_returns = paths_returns[i, :]
        cvar[i] = np.mean(path_returns[path_returns <= var[i]])
        max_drawdown[i] = max_drawdowns(path_returns)

    var = np.mean(var)
    cvar = np.mean(cvar)
    max_drawdown = np.mean(max_drawdown)
    
    if times is not None:
        import matplotlib.pyplot as plt
        # Plot historical data and simulations
        plt.figure(figsize=(15, 6))
        
        paths = paths[:,1:]
        
        # Plot a subset of paths
        num_paths_to_plot = min(100, n_simulations)
        for i in range(num_paths_to_plot):
            plt.plot(times, paths[i], 'b-', alpha=0.1)
        
        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        plt.plot(times, mean_path, 'r-', linewidth=2, label=f'expected')
        
        # Plot confidence intervals
        upper_bound = np.percentile(paths, upper_percentile * 100, axis=0)
        lower_bound = np.percentile(paths, lower_percentile * 100, axis=0)
        plt.plot(times, 
                 upper_bound, 'g--', linewidth=1.5, 
                 label=f'{confidence_level * 100}% Confidence Interval')
        plt.plot(times, 
                 lower_bound, 'g--', linewidth=1.5)
        plt.fill_between(times, 
                         lower_bound, upper_bound, color='g', alpha=0.1)
        
        # Add historical data point (final price)
        plt.plot(times[-1], px_last, 'ko', markersize=8, label='Current Price')
        
        if title in ('', None):
            title = 'GBM MC Simulation'
        plt.title(title)
        plt.ylabel('kUSDc')
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        'expected_return': expected_price/s0 - 1,
        'volatility': np.std(final_prices),
        'var': var,
        'cvar': cvar,
        'max_drawdown': np.mean(max_drawdown)
    }

def bootstrap_backtest(returns, n_samples=1000, sample_length=365):
    """
    Perform bootstrap backtesting by resampling historical returns
    
    Parameters:
        returns (array-like): Historical returns
        n_samples (int): Number of bootstrap samples
        sample_length (int): Length of each bootstrap sample
    
    Returns:
        array: Bootstrapped path statistics
    """
    returns = np.array(returns)
    bootstrap_paths = []
    
    for _ in range(n_samples):
        # Random sampling with replacement
        indices = np.random.randint(0, len(returns), size=sample_length)
        sampled_returns = returns[indices]
        path_return = np.prod(1 + sampled_returns) - 1
        bootstrap_paths.append(path_return)
    
    return np.array(bootstrap_paths)
