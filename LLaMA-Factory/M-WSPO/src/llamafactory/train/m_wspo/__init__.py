# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""m-WSPO training stage.

Implements *m-WSPO: Multimodal Weak-to-Strong Preference Optimization*
(Ishmam, Hossain, Fahim) on top of LLaMA-Factory's multimodal DPO stage.
The unified objective (Eq. 11 of the paper) is

    L_m-WSPO = L_WSPO + alpha * L_img_WSPO + beta_dpo * L_DPO^m
"""

from .workflow import run_m_wspo


__all__ = ["run_m_wspo"]
