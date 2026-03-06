/**
 * Minimal type stubs for @fiftyone/* packages.
 *
 * These packages are provided as UMD globals by the FiftyOne App at runtime
 * and are externalized by vite-plugin-externals at build time.  The stubs
 * here exist solely to keep TypeScript happy during development without
 * requiring a full FiftyOne source checkout.
 */

declare module "@fiftyone/operators" {
  /** Calls a Python panel method via the panel event bus. */
  export function usePanelEvent(): (
    eventName: string,
    options: {
      operator: string;
      params: Record<string, any>;
      callback: (result: any) => void;
    }
  ) => void;

  export interface OperatorExecutor {
    execute: (params: Record<string, any>) => Promise<any>;
    isLoading: boolean;
  }

  /** Executes a registered FiftyOne operator by URI. */
  export function useOperatorExecutor(uri: string): OperatorExecutor;

}

declare module "@fiftyone/state" {
  /** Recoil atom: name of the slice currently active in the modal viewer. */
  export const modalGroupSlice: any;
}

declare module "recoil" {
  export function useRecoilValue<T>(atom: any): T;
}

declare module "@fiftyone/plugins" {
  export enum PluginComponentType {
    Component = "Component",
  }

  export function registerComponent(opts: {
    name: string;
    component: React.ComponentType<any>;
    type: PluginComponentType;
    activator?: (ctx: any) => boolean;
  }): void;
}
