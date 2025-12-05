/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

export default {
  "docs": {
    "About": ["introduction", "design_philosophy", "botorch_and_ax", "papers"],
    "General": ["getting_started"],
    "Basic Concepts": ["overview", "models", "posteriors", "acquisition", "optimization"],
    "Advanced Topics": ["constraints", "objectives", "batching", "samplers"],
    "Multi-Objective Optimization": ["multi_objective"]
  },
  "tutorials": [
    {
      type: 'category',
      label: 'Tutorials',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'tutorials/index',
          label: 'Overview',
        },
        {
          type: 'doc',
          id: 'tutorials/closed_loop_botorch_only/index',
          label: 'Closed Loop BoTorch Only',
        },
      ],
    },
  ],
  "notebooks_community": [
    {
      type: 'category',
      label: 'Community Notebooks',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'notebooks_community/index',
          label: 'Overview',
        },
      ],
    },
  ],
}