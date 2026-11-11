# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == 'math_dapo' or data_source.startswith("aime"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['viki-count']:
        from . import viki_count
        res = viki_count.compute_score(solution_str, ground_truth)
    elif data_source in ['viki_1']:
        from . import viki_1
        res = viki_1.compute_score(solution_str, ground_truth)
    elif data_source in ['viki_2']:
        from . import viki_2
        res = viki_2.compute_score(solution_str, ground_truth)
    elif data_source in ['viki_3']:
        from . import viki_3
        res = viki_3.compute_score(solution_str, ground_truth)
    elif data_source in ['enactgraph']:
        from . import enactgraph
        res = enactgraph.compute_score(solution_str, ground_truth)
    elif data_source in ['lnsdl', 'lnsdl_skip', 'lnsdl_nonskip',
                         'lnsdl_locus', 'lnsdl_locus_s0', 'lnsdl_locus_s1p']:
        # L-NSDL VIKI-L2-train scoped RL (P5.3, DAPO). Needs extra_info to read
        # multi-turn trace signals (stopped_reason / emitted-vs-executed / stacked).
        # The `lnsdl_skip` / `lnsdl_nonskip` variants are ONLY used to tag the
        # fixed probe rows so process_validation_metrics reports the skip and
        # nonskip halves as separate val-aux data_sources (docs/RL_debug.md §4,
        # Problem 4: the probe is 50% skip vs official ~2%, so the aggregate
        # trend conflates learning the upweighted skip family with generalization).
        # The reward math is identical for all three.
        #
        # The `lnsdl_locus*` variants are the same contract over LOCUS-Bench
        # prompts (A0b G1 / P9). Same reward function on purpose: `_r_task`
        # already carries the LOCUS null-plan branch (15.1% of ramp63k prompts
        # prescribe no steps, where the empty plan is the CORRECT answer), and
        # that branch is a verified no-op on VIKI-L2. `_s0` / `_s1p` split the
        # fixed probe by S-axis level so a facet-A collapse cannot hide behind
        # facet-B saturation in the aggregate; the reward math is identical.
        from . import lnsdl
        res = lnsdl.compute_score(solution_str, ground_truth, extra_info=extra_info)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
