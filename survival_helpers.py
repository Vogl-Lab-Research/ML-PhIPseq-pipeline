import logging
from typing import Any, Dict, Optional, Union, List, Tuple

from pathlib import Path

import numpy as np
import pandas as pd

# Plots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ML survival models and utils
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw, concordance_index_censored
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

# Parallelize
from joblib import Parallel, delayed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def c_index_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Custom scorer to calculate Concordance Index (C-index) for single-column `y_true`.
    Negative values in `y_true` indicate censored data; positive values indicate events.

    Parameters:
    - y_true: 1D array of survival times (negative for censored, positive for events).
    - y_pred: Predicted risk scores (higher scores indicate higher risk).

    Returns:
    - C-index: Concordance Index for the predictions.
    """
    # Separate survival times and event indicators
    events = np.where(y_true < 0, 0, 1)  # 0 for censored, 1 for event
    times = np.abs(y_true)  #survival_time=np.abs(y_true) Absolute values for survival times

    y_surv = Surv.from_arrays(event=events, time=times)

    # Calculate and return the C-index
    #return concordance_index_ipcw(y_surv, y_surv, y_pred)[0]
    return concordance_index_censored(y_surv['event'], y_surv['time'], y_pred)[0]


def univariate_cox_score_single(j: int, X: np.ndarray, y_surv) -> float:
    """
    Computes the concordance score for a single feature (column j) using a univariate Cox model.

    Parameters
    ----------
    j : int
        Index of the feature to evaluate.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y_surv : structured array
        Survival data as a structured array (e.g., from Surv.from_arrays).

    Returns
    -------
    float
        The score (e.g., concordance index) for feature j. Returns 0.0 if an error occurs.
    """
    model = CoxPHSurvivalAnalysis()
    try:
        # Extract the j-th feature as a 2D array
        Xj = X[:, j:j+1]
        model.fit(Xj, y_surv)
        score = model.score(Xj, y_surv)
        return score
    except Exception as e:
        #logger.warning(f"Feature index {j} failed with error: {e}")
        return 0.0

def univariate_cox_score(X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> np.ndarray:
    """
    Computes univariate Cox scores for each feature in X in parallel.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        1D array of survival times; negative values indicate censored observations.
    n_jobs : int, optional
        Number of parallel jobs to run (default is -1 to use all available cores).

    Returns
    -------
    np.ndarray
        Array of scores for each feature (shape: (n_features,)).
    """
    # Convert y into a structured survival array: event==1 indicates event occurred,
    # and negative y-values indicate censored observations.
    events = np.where(y < 0, 0, 1)
    times = np.abs(y)
    y_surv = Surv.from_arrays(event=events, time=times)

    n_features = X.shape[1]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(univariate_cox_score_single)(j, X, y_surv) for j in range(n_features)
    )
    return np.array(scores)



# Assuming StratifiedKFoldSurv and c_index_scorer are defined/imported elsewhere

def search_random_best_model(estimator: Any,
                             param_grid: Dict,
                             X_train,
                             y_train,
                             n_splits: int = 5,
                             random_state: int = 420,
                             n_iter: int = 30,
                             n_jobs: int = -1) -> Any:
    """
    Perform randomized hyperparameter search using cross-validation on survival data.

    Parameters
    ----------
    estimator : object
        A scikit-learn estimator (e.g. a survival model).
    param_grid : dict
        Dictionary with parameters names (str) as keys and distributions or lists of parameter settings to try.
    X_train : array-like or DataFrame
        Training data features.
    y_train : array-like or structured array
        Training data survival target.
    n_splits : int, default=5
        Number of splits for StratifiedKFoldSurv.
    random_state : int, default=420
        Seed for reproducibility.
    n_iter : int, default=30
        Number of parameter settings that are sampled in RandomizedSearchCV.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    best_model : object
        The best estimator model found by RandomizedSearchCV.
    """
    # Create a custom scorer based on the C-index
    custom_scorer = make_scorer(c_index_scorer, greater_is_better=True)
    cv = StratifiedKFoldSurv(n_splits=n_splits, random_state=random_state)

    search = RandomizedSearchCV(estimator, param_grid,
                                cv=cv, n_iter=n_iter,
                                scoring=custom_scorer, refit=True,
                                random_state=random_state, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model


def search_grid_best_model(estimator: Any,
                           param_grid: Dict,
                           X_train,
                           y_train,
                           n_splits: int = 5,
                           random_state: int = 420,
                           n_jobs: int = -1) -> Any:
    """
    Perform grid hyperparameter search using cross-validation on survival data.

    Parameters
    ----------
    estimator : object
        A scikit-learn estimator (e.g. a survival model).
    param_grid : dict
        Dictionary with parameters names (str) as keys and lists of parameter settings to try.
    X_train : array-like or DataFrame
        Training data features.
    y_train : array-like or structured array
        Training data survival target.
    n_splits : int, default=5
        Number of splits for StratifiedKFoldSurv.
    random_state : int, default=420
        Seed for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    best_model : object
        The best estimator model found by GridSearchCV.
    """
    # Create a custom scorer based on the C-index
    custom_scorer = make_scorer(c_index_scorer, greater_is_better=True)
    cv = StratifiedKFoldSurv(n_splits=n_splits, random_state=random_state)

    search = GridSearchCV(estimator, param_grid,
                          cv=cv, scoring=custom_scorer, refit=True,
                          n_jobs=n_jobs)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    return best_model



def calculate_time_dependent_auc(y_event_train, y_time_train,
                                 y_event_test, y_time_test, risk_scores,
                                  time_points_highlight=None, num_points=50, buffer=0.001):
    """
     Calculate time-dependent AUC over a set of follow-up time points for the testing set and
     compute the number of samples at risk, censored, and events at each time point.

     Parameters
     ----------
     y_event_train : array-like
         Binary event status for training data (1 if event occurred, 0 if censored).
     y_time_train : array-like
         Survival time for training data.
     y_event_test : array-like
         Binary event status for testing data.
     y_time_test : array-like
         Survival time for testing data.
     risk_scores : array-like
         Predicted risk scores for the testing set.
     time_points_highlight : array-like, optional
         Specific time points to highlight on the AUC curve.
     num_points : int, default 50
         Number of evenly spaced time points for AUC calculation.
     buffer : float, default 0.001
         Small buffer added/subtracted to avoid boundary issues in time calculations.

     Returns
     -------
     auc_values : np.ndarray
         Array of AUC values computed at each time point.
     mean_auc : float
         Mean AUC value across the time points.
     auc_values_highlight : np.ndarray
         AUC values computed specifically at the highlighted time points.
     samples_at_risk : pd.DataFrame
         DataFrame with columns 'At Risk', 'Censored', and 'Events' for each time point.
     time_points : np.ndarray
         The full set of time points used for calculation.
     time_points_highlight : np.ndarray
         The set of highlighted time points.
     """
    # Set up time points within the range of the train and test follow-up times
    valid_indices = y_time_test.abs() <= y_time_train.abs().max()
    y_event_test = y_event_test[valid_indices]
    y_time_test = y_time_test[valid_indices]
    risk_scores = risk_scores[valid_indices]

    min_time = max(y_time_test.abs().min(), y_time_train.abs().min()) + buffer  # Non-negative minimum time
    max_time = y_time_test.abs().max() - buffer  # Ensuring max_time is strictly less than training max

    # Generate evenly spaced time points and merge with highlighted ones
    base_time_points = np.linspace(min_time, max_time, num=num_points)
    if time_points_highlight is None:
        # Default highlighted points: 1 and every 6 time units up to max_time
        time_points_highlight = np.concatenate(([1], np.arange(6, max_time, step=6)))
    # Merge and ensure uniqueness
    time_points = np.unique(np.concatenate((base_time_points, time_points_highlight)))


    # Step 2: Prepare survival data in structured format for sksurv
    y_structured_train = np.array([(e, t) for e, t in zip(y_event_train, y_time_train.abs())],
                                  dtype=[('event', 'bool'), ('time', 'float')])
    y_structured_test = np.array([(e, t) for e, t in zip(y_event_test, y_time_test.abs())],
                                 dtype=[('event', 'bool'), ('time', 'float')])



    # Calculate time-dependent AUC
    auc_values, mean_auc = cumulative_dynamic_auc(y_structured_train, y_structured_test, risk_scores, time_points)
    auc_values_highlight, _ = cumulative_dynamic_auc(y_structured_train, y_structured_test, risk_scores, time_points_highlight)

    # Calculate number of individuals at risk, censored, and events at each time point.
    # We'll use a small epsilon to adjust time boundaries (avoid ties)
    epsilon = 0.01
    at_risk = []
    censored = []
    events = []
    total_samples = len(risk_scores)
    for t in time_points:
        t = t + epsilon
        # Censored: count samples that were censored at time t
        count_censored = np.sum((y_time_test.abs() <= t) & (y_event_test == 0))  # Only censored samples
        # Events: count samples that experienced the event at or before time t
        count_events = np.sum((y_time_test.abs() <= t) & (y_event_test == 1))  # Only event samples

        count_at_risk = total_samples - count_events  - count_censored  # Subtract events and censored

        at_risk.append(count_at_risk)
        censored.append(count_censored)
        events.append(count_events)

    samples_at_risk = pd.DataFrame({
        'At Risk': at_risk,
        'Censored': censored,
        'Events': events
    }, index=time_points)

    return auc_values, mean_auc, auc_values_highlight, samples_at_risk, time_points, time_points_highlight



def calculate_antigen_scores_scaled(
        shap_values: pd.DataFrame,
        y_time: pd.DataFrame,
        y_event: pd.DataFrame,
        top_peptides: list,
        scaler: Optional[MinMaxScaler] = None,
        scaler_antigens: Optional[MinMaxScaler] = None,
        threshold: Optional[float] = None,
        val_quantile: float = 40,
        return_all: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, float, MinMaxScaler, MinMaxScaler]]:
    """
    Calculate antigen scores based on SHAP values for an external dataset.

    The function selects a subset of peptides (top_peptides), scales their SHAP values,
    applies the sign of the original values, and sums the values across features to obtain an antigen score.
    It then scales the antigen scores to the 0-1 range, dichotomizes them using a threshold (default quantile),
    and merges the scores with provided survival time and event DataFrames.

    Parameters
    ----------
    shap_values : pd.DataFrame
        DataFrame of SHAP values (with samples as rows and peptide features as columns).
    y_time : pd.DataFrame
        DataFrame of survival times, indexed by sample.
    y_event : pd.DataFrame
        DataFrame of event statuses (e.g., 1=event, 0=censored), indexed by sample.
    top_peptides : list
        List of peptide column names (subset of shap_values.columns) to be used.
    scaler : MinMaxScaler, optional
        Pre-fitted scaler for SHAP values; if None, a new scaler will be fit.
    scaler_antigens : MinMaxScaler, optional
        Pre-fitted scaler for antigen scores; if None, a new scaler will be fit.
    threshold : float, optional
        Threshold to dichotomize antigen scores. If None, computed as the val_quantile quantile of scaled scores.
    val_quantile : float, default 40
        Quantile (in percent) to use for threshold calculation if threshold is None.
    return_all : bool, default False
        If True, the function returns a tuple containing the antigen scores DataFrame,
        the threshold, the scaler used for SHAP values, and the scaler used for antigen scores.

    Returns
    -------
    antigen_scores_df : pd.DataFrame
        DataFrame with columns: 'Antigen Score', 'Antigen Score (Scaled)', and
        'Antigen Score (Dichotomized)', merged with y_time and y_event.
    If return_all is True, also returns (threshold, scaler, scaler_antigens).

    Raises
    ------
    ValueError
        If top_peptides are not a subset of shap_values.columns.
    """
    # Ensure that shap_values is a DataFrame and top_peptides are valid columns.
    if not isinstance(shap_values, pd.DataFrame):
        raise ValueError("shap_values must be a pandas DataFrame.")
    if not set(top_peptides).issubset(shap_values.columns):
        raise ValueError("Some top_peptides are not present in shap_values.columns.")

    # Work on absolute SHAP values for scaling; then reintroduce sign later.
    abs_shap = abs(shap_values)

    # Scale the SHAP values for the top peptides.
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_shap = pd.DataFrame(scaler.fit_transform(abs_shap[top_peptides]),
                                   index=abs_shap.index) #columns=abs_shap[top_peptides].columns)
        #scaled_shap = scaler.fit_transform(scaled_shap)

    else:
        scaled_shap = pd.DataFrame(scaler.transform(abs_shap[top_peptides]),
                                   index=abs_shap.index) #columns=abs_shap[top_peptides].columns)
        #scaled_shap = pd.DataFrame(scaler.transform(scaled_shap), index=scaled_shap.index)
        scaled_shap = np.clip(scaled_shap, 0, 1)

    # Reintroduce the original sign for each value.
    # We assume the original shap_values DataFrame has the same structure.
    # scaled_shap.columns = shap_values.columns
    # scaled_shap = scaled_shap[top_peptides]
    scaled_shap.columns = shap_values[top_peptides].columns
    signed_shap = scaled_shap * np.sign(shap_values[top_peptides])

    # Sum across features to produce the antigen score.
    antigen_scores = signed_shap.sum(axis=1).to_frame(name='Antigen Score')

    # Scale the antigen scores.
    if scaler_antigens is None:
        scaler_antigens = MinMaxScaler(feature_range=(0, 1))
        antigen_scores_scaled = pd.DataFrame(scaler_antigens.fit_transform(antigen_scores),
                                             index=antigen_scores.index,
                                             columns=['Antigen Score (Scaled)'])
    else:
        antigen_scores_scaled = pd.DataFrame(scaler_antigens.transform(antigen_scores),
                                             index=antigen_scores.index,
                                             columns=['Antigen Score (Scaled)'])
        antigen_scores_scaled = np.clip(antigen_scores_scaled, 0, 1)

    # Determine the threshold to dichotomize the scores.
    if threshold is None:
        threshold = antigen_scores_scaled.quantile(val_quantile / 100) #antigen_scores_scaled.quantile(val_quantile / 100).iloc[0]

    antigen_scores_dichotomized = (antigen_scores_scaled >= threshold).astype(int)
    antigen_scores_dichotomized.rename(columns={'Antigen Score (Scaled)': 'Antigen Score (Dichotomized)'}, inplace=True)

    #antigen_scores = antigen_scores.rename(columns={antigen_scores.columns[0]: 'Antigen Score'})
    #antigen_scores_scaled = antigen_scores_scaled.rename(columns={antigen_scores_scaled.columns[0]: 'Antigen Score (Scaled)'})
    #antigen_scores_dichotomized = antigen_scores_dichotomized.rename(columns={antigen_scores_dichotomized.columns[0]: 'Antigen Score (Dichotomized)'})

    # Merge with survival data (y_time and y_event)
    merged_df = y_time.merge(y_event, left_index=True, right_index=True)
    merged_df = merged_df.merge(antigen_scores, left_index=True, right_index=True)
    merged_df = merged_df.merge(antigen_scores_scaled, left_index=True, right_index=True)
    antigen_scores_df = merged_df.merge(antigen_scores_dichotomized, left_index=True, right_index=True)

    if return_all:
        return antigen_scores_df, threshold, scaler, scaler_antigens
    else:
        return antigen_scores_df


def perform_logrank_test(df: pd.DataFrame,
                         time_column: str,
                         event_column: str,
                         group_column: str) -> float:
    """
    Perform a log-rank test between two groups defined by the group_column (expected values: 0 and 1).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the survival data.
    time_column : str
        Name of the column containing the survival times.
    event_column : str
        Name of the column indicating the event occurrence (1=event, 0=censored).
    group_column : str
        Name of the column indicating the group assignment (0 for one group, 1 for the other).

    Returns
    -------
    float
        The p-value from the log-rank test.

    Raises
    ------
    ValueError
        If the group_column does not contain exactly two groups (e.g. 0 and 1).
    """
    # Verify that the group_column contains exactly two unique groups
    groups = np.sort(df[group_column].unique())
    if len(groups) != 2:
        raise ValueError(f"Expected two groups in column '{group_column}', but found: {groups}")

    # Split the DataFrame into the two groups
    group0 = df[df[group_column] == groups[0]]
    group1 = df[df[group_column] == groups[1]]

    # Perform the log-rank test
    results = logrank_test(group0[time_column],
                           group1[time_column],
                           event_observed_A=group0[event_column],
                           event_observed_B=group1[event_column])

    p_value = results.p_value
    logger.info(f"Log-Rank Test p-value: {p_value:.4f}")

    return p_value


def align_features(X_train, X_ext):
    """Align features between training and external test sets."""
    common_features = list(set(X_train.columns).intersection(X_ext.columns))
    if common_features:
        X_train = X_train[common_features]
        X_ext = X_ext[common_features]
        return X_train, X_ext
    else:
        raise ValueError(f"Training and External set shared no common features")


""" Plots """

# Example usage (use real data for these variables):
# plot_time_dependent_auc(time_points, auc_values, mean_auc, time_points_highlight=time_points_highlight, auc_values_highlight=auc_values_highlight)


def plot_time_dependent_auc(
        time_points: Union[list, np.ndarray],
        auc_values: Union[list, np.ndarray],
        mean_auc: float,
        samples_at_risk_df: pd.DataFrame,
        time_points_highlight: Optional[Union[list, np.ndarray]] = None,
        auc_values_highlight: Optional[Union[list, np.ndarray]] = None,
        max_time_point: Optional[int] = None,
        time_measure: str = 'Months',
        color_auc: str = 'dodgerblue',
        suffix_file: Optional[str] = None,
        figures_dir: str = './',
        save_fig: bool = False) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Plots time-dependent AUC over follow-up time with a table displaying samples at risk, censored, and events.
    The table is placed below the AUC plot as a horizontal table aligned with the x-axis.

    Parameters
    ----------
    time_points : array-like
        The time points (e.g., months) to plot on the x-axis.
    auc_values : array-like
        The AUC values corresponding to each time point.
    mean_auc : float
        The mean AUC value to be displayed as a horizontal line.
    samples_at_risk_df : pd.DataFrame
        DataFrame containing 'At Risk', 'Censored', and 'Events' per time point. Its index should
        correspond to time_points.
    time_points_highlight : array-like, optional
        Specific time points to highlight on the AUC curve.
    auc_values_highlight : array-like, optional
        AUC values at the specific points to highlight.
    max_time_point: int, optional
        Maximum time point to plot. If not provided, all time points will be plotted.
    time_measure : str, default 'Months'
        Label for the time unit.
    color_auc : str, default 'dodgerblue'
        Color used for the AUC line.
    suffix_file : str, optional
        Suffix to add to the saved figure filename.
    figures_dir : str, default './'
        Directory where the figure will be saved.
    save_fig : bool, default False
        Whether to save the figure to file.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax_auc : matplotlib.axes.Axes
        The axis containing the AUC plot.
    ax_table : matplotlib.axes.Axes
        The axis containing the samples at risk table.
    """
    # Create a figure with GridSpec to control height ratios
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[0.8, 0.15], figure=fig)
    ax_auc = fig.add_subplot(gs[0])

    # Plot AUC curve and mean AUC line
    ax_auc.plot(time_points, auc_values, color=color_auc, linestyle='-', linewidth=2, label='Time-Dependent AUC')
    ax_auc.axhline(y=mean_auc, color='orange', linestyle='--', linewidth=1.5, label=f'Mean AUC = {mean_auc:.3f}')
    ax_auc.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5)

    # Highlight specific time points if provided
    if time_points_highlight is not None and auc_values_highlight is not None:
        ax_auc.scatter(time_points_highlight, auc_values_highlight, color=color_auc, s=50, zorder=3)

    ax_auc.set_xlabel(f'Time ({time_measure})', fontsize=12)
    ax_auc.set_ylabel('Time-Dependent AUC', fontsize=12)
    ax_auc.set_ylim([0.0, 1.05])
    if max_time_point is not None:
        ax_auc.set_xlim([0, max_time_point])  # Ensure x-axis covers max time point given
    else:
        ax_auc.set_xlim([0, max(time_points)])  # Ensure x-axis covers all time points

    ax_auc.legend(fontsize=10, loc='lower right')
    ax_auc.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    # Bottom panel: Table for samples at risk, censored, and events
    ax_table = fig.add_subplot(gs[1])
    ax_table.set_xticks(time_points)
    ax_table.set_xticklabels([])  # No tick labels on x-axis
    if max_time_point is not None:
        ax_table.set_xlim([0, max_time_point])  # Ensure x-axis covers max time point given
    else:
        ax_table.set_xlim([0, max(time_points)])  # Ensure x-axis covers all time points


    # Set y-tick positions and labels for the table (order: Events, Censored, At Risk)
    y_positions = np.array([0.2,0.5,0.8])
    ax_table.set_yticks(y_positions)
    ax_table.set_yticklabels(['Events', 'Censored', 'At Risk'], ha='right', fontsize=10)
    ax_table.grid(False)
    ax_table.set_facecolor('white')

    # If time_points_highlight is provided, ensure max time point is included
    if time_points_highlight is not None:
        tp_high = list(time_points_highlight)
        if max_time_point is None:
            max_time_point = max(time_points)
        if max_time_point not in tp_high:
            tp_high.append(max_time_point)
    else:
        tp_high = time_points  # If no highlight provided, use all time points

    # Iterate over each highlighted time point and place text for table values
    for time_point in tp_high:
        try:
            at_risk_val = samples_at_risk_df.loc[time_point, 'At Risk']
            censored_val = samples_at_risk_df.loc[time_point, 'Censored']
            events_val = samples_at_risk_df.loc[time_point, 'Events']
        except KeyError:
            # Skip time points not found in the DataFrame
            continue

        ax_table.text(time_point, 0.2, f'{events_val}', ha='center', va='center', fontsize=10, color='black')
        ax_table.text(time_point, 0.5, f'{censored_val}', ha='center', va='center', fontsize=10, color='black')
        ax_table.text(time_point, 0.8, f'{at_risk_val}', ha='center', va='center', fontsize=10, color='black')


    #plt.tight_layout()

    # Save the figure if requested
    if save_fig:
        if suffix_file is None:
            suffix_file = "default"
        save_path = Path(figures_dir) / f'time-dependent_auc_{suffix_file}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax_auc, ax_table




logger = logging.getLogger(__name__)


def plot_kaplan_meier(df: pd.DataFrame,
                      time_column: str,
                      event_column: str,
                      group_column: str,
                      labels: Dict[Union[int, str], str],
                      ax: Optional[plt.Axes] = None,
                      title: str = '',
                      xlabel: str = '',
                      ylabel: str = '',
                      suffix_file = None,
                      save_fig: bool = False,
                      figures_dir: str = './',
                      ) -> plt.Axes:
    """
    Create a Kaplan-Meier survival plot with log-rank test p-value annotation and at-risk counts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing survival data.
    time_column : str
        Name of the column containing survival times.
    event_column : str
        Name of the column indicating event occurrence (1 = event, 0 = censored).
    group_column : str
        Name of the column indicating the group (expected to have exactly two unique values).
    labels : dict
        Dictionary mapping group values (keys) to display labels (values).
        The keys should correspond to the values in group_column.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the plot. If None, a new figure and axes are created.
    title : str, optional
        Title for the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    suffix_file : str, optional default None
        Suffix used in the filename when saving.
    save_fig : bool, default False
        If True, saves the figure to file.
    figures_dir : str, default './'
        Directory in which to save the figure.


    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the Kaplan-Meier plot.

    Raises
    ------
    ValueError
        If the group_column does not contain exactly two unique groups.
    """
    # Ensure there are exactly two groups.
    unique_groups = np.sort(df[group_column].unique())
    if len(unique_groups) != 2:
        raise ValueError(f"Expected exactly two groups in '{group_column}', but found: {unique_groups}")

    # Define colors explicitly for each group (or you can extend this mapping if needed)
    group_colors = {unique_groups[0]: 'dodgerblue', unique_groups[1]: 'darkorange'}

    # Create figure and axis if not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    kmf_list = []
    # Plot survival functions in order based on the sorted unique groups
    for group_value in unique_groups:
        mask = df[group_column] == group_value
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_column][mask], event_observed=df[event_column][mask], label=labels[group_value])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=group_colors[group_value])
        kmf_list.append(kmf)

    # Perform log-rank test between the two groups
    group1 = df[df[group_column] == unique_groups[0]]
    group2 = df[df[group_column] == unique_groups[1]]
    if not group1.empty and not group2.empty:
        results = logrank_test(group1[time_column], group2[time_column],
                               event_observed_A=group1[event_column],
                               event_observed_B=group2[event_column])
        p_value = results.p_value
        p_value_text = f'p = {p_value:.3f}'
        ax.text(0.7, 0.05, p_value_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.5))
    else:
        logger.warning("One of the groups is empty; log-rank test not performed.")

    # Add at-risk counts to the plot (from lifelines)
    if len(kmf_list) >= 2:
        add_at_risk_counts(kmf_list[0], kmf_list[1], ax=ax)
    else:
        logger.warning("At-risk counts could not be added because fewer than two groups were found.")

    # Customize axes
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=8)

    fig.tight_layout()

    # Save figure if required
    if save_fig:
        if suffix_file is None:
            suffix_file = "default"
        save_path = Path(figures_dir) / f'kaplan_meier_{suffix_file}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {save_path}")

    return ax

# Classes

class CoxnetWrapper(CoxnetSurvivalAnalysis):
    """
    A wrapper for CoxnetSurvivalAnalysis that automatically converts raw survival data
    (with negative values indicating censoring) into a structured array and selects
    coefficients corresponding to the final (smallest) alpha.
    """
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Coxnet model to X and y. If y is not already structured, it is converted.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            1D array of survival times; negative values indicate censoring, positive values indicate event.

        Returns
        -------
        self : CoxnetWrapper
            The fitted model.
        """
        # Convert y to structured format if needed
        if not (hasattr(y, "dtype") and y.dtype.names is not None):
            events = np.where(y < 0, 0, 1)
            times = np.abs(y)
            y = Surv.from_arrays(event=events, time=times)
        # Call the parent class's fit method
        super().fit(X, y)
        #If self.coef_ has multiple columns (one per alpha), choose the coefficients for the last alpha
        #if self.coef_.ndim > 1:
        #logger.info(f"Coefficient shape before selection: {self.coef_.shape}")
        self.coef_ = self.coef_[:, -1]

        return self


class StratifiedKFoldSurv:
    def __init__(self, n_splits=10, shuffle=True, random_state=420):
        self.n_splits = n_splits
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y, groups=None):
        # Define labels based on the sign of y (negative for censored, positive for deceased)

        if isinstance(y, np.ndarray):  # y is a numpy array from Surv.from_arrays
            labels = y['event']  # Event status is the first column (event = 1, censored = 0)
        else:
            labels = np.where(y < 0, 0, 1)  # 0 for censored, 1 for deceased

        # Stratify based on these labels
        return self.skf.split(X, labels)

    def get_n_splits(self, X=None, y=None, groups=None):
        # We don't need to use the `groups` argument here, just return `n_splits`
        return self.n_splits


