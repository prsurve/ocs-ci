# -*- coding: utf-8 -*-
"""
Pytest configuration for unit_tests/.

Installs the custom log record factory so that all log records carry the
'clusterctx' field required by the project-wide log format in pytest.ini.
"""
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_log_record_factory():
    """
    Add 'clusterctx' to every log record to satisfy the pytest.ini log format.
    Mirrors ocs_ci/framework/tests/conftest.py.
    """
    from ocs_ci.framework.logger_factory import set_log_record_factory

    set_log_record_factory()
