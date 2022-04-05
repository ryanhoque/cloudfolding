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

"""Support action template related functionalities.

An action template is a pre-defined sequence of commands which can be sent
to a robot for execution.

An action template command can be static. For example, moving robot to a
fixed pose or turning on/off vacuum.

An action template command can also be parameterized by user inputs.
For example, a pick action takes a pick point as input. A pick point can be
generated by a human operator, when clicking an object in the Reach UI.
A pick point can also be generated by a ML model based on a camera image.

PyReach only provides bare minimum support for the action template right now.
A client can retrieve a list of predefined action template names, and use
the name with the Arm.execute_action and Arm.async_execute_action functions.
"""
from typing import Tuple


class Actions(object):
  """Interface for accessing action template related functionalities."""

  def action_names(self) -> Tuple[str, ...]:
    """Return the list of available action template names."""
    raise NotImplementedError