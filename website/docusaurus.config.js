/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

module.exports={
  "title": "BoTorch",
  "tagline": "Bayesian Optimization in PyTorch",
  "url": "https://botorch.org",
  "baseUrl": "/",
  "organizationName": "pytorch",
  "projectName": "botorch",
  "scripts": [
    "/js/code_block_buttons.js",
    "https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js",
  ],
  "markdown": {
    format: "detect"
  },
  "stylesheets": [
    "/css/code_block_buttons.css",
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
  "favicon": "img/botorch.ico",
  "customFields": {
    "users": [],
    "wrapPagesHTML": true
  },
  "onBrokenLinks": "throw",
  "onBrokenMarkdownLinks": "warn",
  "presets": [
    [
      "@docusaurus/preset-classic",
      {
        "docs": {
          "showLastUpdateAuthor": true,
          "showLastUpdateTime": true,
          "editUrl": "https://github.com/meta-pytorch/botorch/edit/main/docs/",
          "path": "../docs",
          "sidebarPath": "../website/sidebars.js",
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          exclude: [
            "**/tutorials/bope/**",
            "**/tutorials/turbo_1/**", 
            "**/tutorials/baxus/**",
            "**/tutorials/scalable_constrained_bo/**",
            "**/tutorials/saasbo/**",
            "**/tutorials/cost_aware_bayesian_optimization/**",
            "**/tutorials/Multi_objective_multi_fidelity_BO/**",
            "**/tutorials/bo_with_warped_gp/**",
            "**/tutorials/thompson_sampling/**",
            "**/tutorials/ibnn_bo/**",
            "**/tutorials/custom_model/**",
            "**/tutorials/multi_objective_bo/**",
            "**/tutorials/constrained_multi_objective_bo/**",
            "**/tutorials/robust_multi_objective_bo/**",
            "**/tutorials/decoupled_mobo/**",
            "**/tutorials/custom_acquisition/**",
            "**/tutorials/fit_model_with_torch_optimizer/**",
            "**/tutorials/compare_mc_analytic_acquisition/**",
            "**/tutorials/optimize_with_cmaes/**",
            "**/tutorials/optimize_stochastic/**",
            "**/tutorials/batch_mode_cross_validation/**",
            "**/tutorials/one_shot_kg/**",
            "**/tutorials/max_value_entropy/**",
            "**/tutorials/GIBBON_for_efficient_batch_entropy_search/**",
            "**/tutorials/risk_averse_bo_with_environmental_variables/**",
            "**/tutorials/risk_averse_bo_with_input_perturbations/**",
            "**/tutorials/constraint_active_search/**",
            "**/tutorials/information_theoretic_acquisition_functions/**",
            "**/tutorials/relevance_pursuit_robust_regression/**",
            "**/tutorials/meta_learning_with_rgpe/**",
            "**/tutorials/vae_mnist/**",
            "**/tutorials/multi_fidelity_bo/**",
            "**/tutorials/discrete_multi_fidelity_bo/**",
            "**/tutorials/composite_bo_with_hogp/**",
            "**/tutorials/composite_mtbo/**",
            "**/notebooks_community/clf_constrained_bo/**",
            "**/notebooks_community/hentropy_search/**",
            "**/notebooks_community/multi_source_bo/**",
            "**/notebooks_community/vbll_thompson_sampling/**"
          ],
        },
        "blog": {},
        "theme": {
          "customCss": "static/css/custom.css"
        },
        "gtag": {
          "trackingID": "G-CXN3PGE3CC"
        }
      }
    ]
  ],
  "plugins": [],
  "themeConfig": {
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
    "navbar": {
      "title": "BoTorch",
      "logo": {
        "src": "img/botorch.png"
      },
      "items": [
        {
          "type": "docSidebar",
          "sidebarId": "docs",
          "label": "Docs",
          "position": "left"
        },
        {
          "type": "docSidebar",
          "sidebarId": "tutorials",
          "label": "Tutorials",
          "position": "left"
        },
        {
          "type": "custom-docSidebar",
          "sidebarId": "notebooks_community",
          "label": "Community Notebooks",
          "position": "left"
        },
        {
          "href": "https://botorch.readthedocs.io/",
          "label": "API Reference",
          "position": "left",
          "target": "_blank",
        },
        {
          "href": "/docs/papers",
          "label": "Papers",
          "position": "left"
        },
        {
          type: 'docsVersionDropdown',
          position: 'right',
          dropdownItemsAfter: [
              {
                type: 'html',
                value: '<hr class="margin-vert--sm">',
              },
              {
                type: 'html',
                className: 'margin-horiz--sm text--bold',
                value: '<small>Archived versions<small>',
              },
              {
                href: 'https://archive.botorch.org/versions',
                label: '<= 0.12.0',
              },
            ],
        },
        {
          "href": "https://github.com/meta-pytorch/botorch",
          "className": "header-github-link",
          "aria-label": "GitHub",
          "position": "right"
        },
      ]
    },
    "image": "img/botorch.png",
    "footer": {
      style: 'dark',
      "logo": {
        alt: "Botorch",
        "src": "img/meta_opensource_logo_negative.svg",
      },
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: 'docs/introduction',
            },
            {
              label: 'Getting Started',
              to: 'docs/getting_started',
            },
            {
              label: 'Tutorials',
              to: 'docs/tutorials/',
            },
            {
              label: 'API Reference',
              to: 'https://botorch.readthedocs.io/',
            },
            {
              label: 'Paper',
              href: 'https://arxiv.org/abs/1910.06403',
            },
          ],
        },
        {
          title: 'Legal',
          // Please do not remove the privacy and terms, it's a legal requirement.
          items: [
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
              target: '_blank',
              rel: 'noreferrer noopener',
            },
          ],
        },
        {
          title: 'Social',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/meta-pytorch/botorch',
            },
            {
              html: `<iframe
                src="https://ghbtns.com/github-btn.html?user=pytorch&amp;repo=botorch&amp;type=star&amp;count=true&amp;size=small"
                title="GitHub Stars"
              />`,
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc.`,
    },
    "algolia": {
      "appId": "ASRH08QMIJ",
      "apiKey": "e5edacd85d22b57ef7ca2d06ba6333f8",
      "indexName": "botorch"
    }
  }
}
