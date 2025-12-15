import logging
import pytest

from ocs_ci.framework import config
from ocs_ci.framework.pytest_customization.marks import (
    acceptance,
    ignore_owner,
    tier1,
    tier2,
    tier3,
    tier4a,
    tier4b,
    tier4c,
    pre_upgrade,
    post_upgrade,
    pre_ocs_upgrade,
    pre_ocp_upgrade,
    post_ocp_upgrade,
    post_ocs_upgrade,
    workloads,
    performance,
    scale,
    ocs_ci_utility,
)
from ocs_ci.utility.ssl_certs import configure_custom_ingress_cert

log = logging.getLogger(__name__)


@ignore_owner
@tier1
@acceptance
@tier2
@tier3
@tier4a
@tier4b
@tier4c
@pre_upgrade
@post_upgrade
@pre_ocs_upgrade
@pre_ocp_upgrade
@post_ocp_upgrade
@post_ocs_upgrade
@workloads
@performance
@scale
@ocs_ci_utility
class TestFailurePropagator:
    """
    Test class for failure propagator test case
    """

    def test_failure_propagator(self):
        """
        This test intention is to run last and propagate teardown failures caught during the test execution,
        so regular test cases won't false negatively fail
        """
        configure_custom_ingress_cert()
