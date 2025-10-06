# Darwinacci-Ω Prometheus/Grafana quick start

- Exporter: set `DARWINACCI_PROMETHEUS=1` and optionally `DARWINACCI_PROM_PORT` (default 8011)
- Key gauges: `darwinacci_best_score`, `darwinacci_coverage`, `darwinacci_novelty_size`, `darwinacci_mut`, `darwinacci_cx`, `darwinacci_elite`, `darwinacci_z_best`, `darwinacci_z_cov`, `darwinacci_accepted_rate`, `darwinacci_canary_pass_rate`

Example PromQL:
- avg_over_time(darwinacci_best_score[5m])
- min_over_time(darwinacci_coverage[2h])
- darwinacci_z_cov > 3
- darwinacci_canary_pass_rate / clamp_min(darwinacci_accepted_rate, 1e-9)

Grafana:
- Import `monitoring/grafana_dashboard.json` and set Prometheus datasource.
