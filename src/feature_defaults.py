"""
Shared feature defaults for every script that builds a betise scenario.

Lives in its own module so generate.py and validate_config.py read one copy. A
second copy would drift: the damped_trend key was missing from this dict for the
whole 0.4.0 -> 0.5.0 upgrade precisely because nothing compared it to betise.
"""

# Every overlay feature in betise's FEATURE_ORDER (0.5.0 and 0.6.0 have the same 15), all off. Scenarios switch on
# only what they need, so a feature left out of a scenario is explicitly disabled
# rather than left at whatever load_config defaults to.
#
# Keep this in step with betise's FEATURE_ORDER on every upgrade — validate_config.py
# checks it and fails if a key is missing. damped_trend is new in 0.5.0; it defaults to
# off there, so leaving it out changed nothing (checked by generating with and without
# it — identical series), but the point of this dict is not to depend on the defaults.
#
# Seasonality is deliberately absent: it is a property of base_series
# (single_seasonality / multiple_seasonality / sarma / sarima), not an overlay
# feature, and betise 0.5.0 codifies that in core/rules.py. Listing it here would be
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
