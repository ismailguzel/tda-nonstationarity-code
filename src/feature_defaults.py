"""
Shared feature defaults for every script that builds a betise scenario.

Lives in its own module so generate.py and validate_config.py read one copy.
"""

# Every overlay feature in betise's FEATURE_ORDER (15 in betise 0.6.0), all off. Scenarios switch on
# only what they need, so a feature left out of a scenario is explicitly disabled
# rather than left at whatever load_config defaults to.
#
# validate_config.py checks this dict against betise's FEATURE_ORDER and fails if a key
# is missing, so that no feature depends on the defaults of load_config.
#
# Seasonality is deliberately absent: it is a property of base_series
# (single_seasonality / multiple_seasonality / sarma / sarima), not an overlay
# feature, and betise codifies that in core/rules.py. Listing it here would be
# a dead key.
ALL_FEATURES_OFF = {
    "linear_trend": {"enabled": False},
    "quadratic_trend": {"enabled": False},
    "cubic_trend": {"enabled": False},
    "exponential_trend": {"enabled": False},
    "damped_trend": {"enabled": False},
    "arch": {"enabled": False},
    "garch": {"enabled": False},
    "egarch": {"enabled": False},
    "aparch": {"enabled": False},
    "mean_shift": {"enabled": False},
    "variance_shift": {"enabled": False},
    "trend_shift": {"enabled": False},
    "point_anomaly": {"enabled": False},
    "collective_anomaly": {"enabled": False},
    "contextual_anomaly": {"enabled": False},
}
