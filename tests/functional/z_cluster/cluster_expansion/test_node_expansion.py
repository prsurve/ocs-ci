import logging
import pytest

from ocs_ci.deployment.zones import create_dummy_zone_labels
from ocs_ci.framework.testlib import tier1, ignore_leftovers, ManageTest
from ocs_ci.ocs import constants
from ocs_ci.ocs.cluster import CephCluster
from ocs_ci.framework.pytest_customization.marks import (
    skipif_flexy_deployment,
    skipif_ibm_flash,
    skipif_managed_service,
    skipif_hci_provider_and_client,
    brown_squad,
)
from ocs_ci.ocs.node import get_nodes, add_disk_to_node
from ocs_ci.ocs.resources.storage_cluster import (
    in_transit_encryption_verification,
    set_in_transit_encryption,
    get_in_transit_encryption_config_state,
)
from ocs_ci.framework import config

logger = logging.getLogger(__name__)


@brown_squad
# https://github.com/red-hat-storage/ocs-ci/issues/4802
@skipif_managed_service
@skipif_hci_provider_and_client
@skipif_flexy_deployment
@skipif_ibm_flash
@ignore_leftovers
@tier1
class TestAddNode(ManageTest):
    """
    Automates adding worker nodes to the cluster while IOs
    """


    def test_add_ocs_node(self):
        """
        Test to add ocs nodes and wait till rebalance is completed.

        Following operations will be performed after adding node to the cluster.
        1. Enable intransit encryption and verify.
        2. Disable intransit encryption and verify.

        """
        create_dummy_zone_labels()
