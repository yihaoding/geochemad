"""Vendored subset of geo_dataset our_model (feature_extractor only).
Heavy checkpoints/outputs/logs are intentionally excluded; extract_hidden_state
loads a checkpoint only if actually called (not on the T3 recon-head path)."""
