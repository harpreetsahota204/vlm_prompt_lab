import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import VlmPromptLabPanel from "./VlmPromptLabPanel";

/**
 * Register the React panel component.
 *
 * The ``name`` must match the ``component`` kwarg passed to
 * ``types.View(component="VlmPromptLabPanel", ...)`` in the Python
 * ``VlmPromptLabPanel.render()`` method.
 */
registerComponent({
  name: "VlmPromptLabPanel",
  component: VlmPromptLabPanel,
  type: PluginComponentType.Component,
});
