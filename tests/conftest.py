import hypothesis as hyp

hyp.settings.register_profile("allow-slow", hyp.settings(
    suppress_health_check=[hyp.HealthCheck.too_slow]
))
