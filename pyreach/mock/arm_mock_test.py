# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ArmMock."""

import unittest

from pyreach import arm
from pyreach.gyms import arm_element
from pyreach.mock import arm_mock


class TestArmMock(unittest.TestCase):
  """Test the TestLogger."""

  def test_arm(self) -> None:
    """Test the MockLogger."""
    arm_config: arm_element.ReachArm = arm_element.ReachArm("arm")
    mock_arm: arm_mock.ArmMock = arm_mock.ArmMock(arm_config)
    assert isinstance(mock_arm, arm.Arm)
    assert isinstance(mock_arm, arm_mock.ArmMock)


if __name__ == "__main__":
  unittest.main()
