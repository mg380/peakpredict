"""Shared contracts: config/secrets, logging, event maps, schemas, normalization, IO.

This package is imported by the scraper, pipeline, and dashboard components. The
normalization function and the upload/feature schemas defined here are the single
source of truth — the pipeline trains against them and the dashboard validates
uploads against them, so the two can never diverge.
"""
