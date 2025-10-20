/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from "react";
import { useDocsVersionCandidates } from '@docusaurus/plugin-content-docs/client';
import DocSidebarNavbarItem from '@theme-original/NavbarItem/DocSidebarNavbarItem';


/**
 * Custom implementation of DocSidebarNavbarItem that only renders if the
 * sidebar exists in the current version context.
 *
 * DocSidebarNavbarItem assumes that the provided sidebarId exists in every
 * version of the docs but this is not true in our case since we have added new
 * sidebars not present in previous versions.
 */
export default function ConditionalDocSidebarNavbarItem(props) {
  const docsVersionCandidates = useDocsVersionCandidates(props.docsPluginId);
  if (docsVersionCandidates?.length > 0 && !docsVersionCandidates[0]?.sidebars?.hasOwnProperty(props.sidebarId)) {
    return null;
  }
  return <DocSidebarNavbarItem {...props} />;
}
