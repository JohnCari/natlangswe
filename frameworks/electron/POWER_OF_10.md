# The Power of 10 Rules for Safety-Critical Electron Code

## Background

The **Power of 10 Rules** were created in 2006 by **Gerard J. Holzmann** at NASA's Jet Propulsion Laboratory (JPL) Laboratory for Reliable Software. These rules were designed for writing safety-critical code in C that could be effectively analyzed by static analysis tools.

The rules were incorporated into JPL's institutional coding standard and used for major missions including the **Mars Science Laboratory** (Curiosity Rover, 2012).

> *"If these rules seem draconian at first, bear in mind that they are meant to make it possible to check safety-critical code where human lives can very literally depend on its correctness."* — Gerard Holzmann

**Note:** Electron apps run with full OS-level privileges. Unlike browser-only web apps, a vulnerability in Electron can give an attacker access to the filesystem, shell, and native APIs. These rules treat every IPC boundary as a security perimeter.

---

## The Original 10 Rules (C Language)

| # | Rule |
|---|------|
| 1 | Restrict all code to very simple control flow constructs—no `goto`, `setjmp`, `longjmp`, or recursion |
| 2 | Give all loops a fixed upper bound provable by static analysis |
| 3 | Do not use dynamic memory allocation after initialization |
| 4 | No function longer than one printed page (~60 lines) |
| 5 | Assertion density: minimum 2 assertions per function |
| 6 | Declare all data objects at the smallest possible scope |
| 7 | Check all return values and validate all function parameters |
| 8 | Limit preprocessor to header includes and simple macros |
| 9 | Restrict pointers: single dereference only, no function pointers |
| 10 | Compile with all warnings enabled; use static analyzers daily |

---

## The Power of 10 Rules — Electron Edition

### Rule 1: Simple Control Flow — No Recursive IPC Chains, Guard Clauses

**Original Intent:** Eliminate complex control flow that impedes static analysis and can cause stack overflows.

**Electron Adaptation:**

```typescript
// BAD: Recursive IPC ping-pong between main and renderer
// main.ts
ipcMain.handle('process-data', async (event, data) => {
  const partial = transform(data);
  // Sends back to renderer, which calls 'process-data' again — unbounded recursion
  event.sender.send('need-more', partial);
});

// renderer.ts
window.electronAPI.onNeedMore((partial) => {
  const next = prepareNext(partial);
  window.electronAPI.processData(next);  // Calls main again — infinite loop risk
});

// BAD: Deeply nested conditionals in IPC handler
ipcMain.handle('save-file', async (event, data) => {
  if (event.senderFrame) {
    if (event.senderFrame.url) {
      if (data) {
        if (data.path) {
          if (data.content) {
            await fs.writeFile(data.path, data.content);
            return { success: true };
          }
        }
      }
    }
  }
  return { success: false };
});

// GOOD: Single-direction IPC flow — renderer requests, main responds
// main.ts
ipcMain.handle('process-data', async (event, rawData: unknown) => {
  const data = DataSchema.parse(rawData);

  const result = await processAllSteps(data);  // Main orchestrates everything
  return { success: true, data: result };
});

// renderer.ts
const result = await window.api.processData(inputData);
renderResult(result);  // No callback to main — flow ends here

// GOOD: Guard clauses with early returns
ipcMain.handle('save-file', async (event, rawData: unknown) => {
  if (!event.senderFrame?.url) return { success: false, error: 'No sender' };

  const senderUrl = new URL(event.senderFrame.url);
  if (senderUrl.protocol !== 'app:') return { success: false, error: 'Untrusted sender' };

  const result = SaveFileSchema.safeParse(rawData);
  if (!result.success) return { success: false, error: 'Invalid data' };

  await fs.writeFile(result.data.path, result.data.content);
  return { success: true };
});
```

**Guidelines:**
- No recursive IPC message chains between main and renderer
- Use guard clauses in `ipcMain.handle` callbacks — validate and return early
- Maximum 3-4 levels of nesting in any handler
- Single-direction data flow: renderer requests via `invoke`, main responds directly
- Never have the main process call back to the renderer to continue a chain

---

### Rule 2: Bounded Loops — Bounded Retries, Paginate IPC Data

**Original Intent:** Ensure all loops terminate and can be analyzed statically.

**Electron Adaptation:**

```typescript
// BAD: Unbounded retry loop
async function connectToService(): Promise<Connection> {
  while (true) {
    try {
      return await Service.connect();
    } catch {
      await delay(1000);  // Retries forever
    }
  }
}

// BAD: Sending unbounded data over IPC
ipcMain.handle('get-all-logs', async () => {
  return await db.logs.findMany();  // Could be millions of rows
});

// BAD: Unbounded window iteration
function broadcastToAll(channel: string, data: unknown) {
  BrowserWindow.getAllWindows().forEach(win => {
    win.webContents.send(channel, data);  // Unknown number of windows
  });
}

// GOOD: Bounded retry with exponential backoff
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 1000;

async function connectToService(): Promise<Connection> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return await Service.connect();
    } catch (error) {
      if (attempt === MAX_RETRIES - 1) throw error;
      await delay(BASE_DELAY_MS * Math.pow(2, attempt));
    }
  }

  throw new Error('Unreachable');
}

// GOOD: Paginated IPC data
const PAGE_SIZE = 100;
const MAX_PAGES = 1000;

ipcMain.handle('get-logs', async (event, rawParams: unknown) => {
  validateSender(event);
  const { page } = PaginationSchema.parse(rawParams);
  const safePage = Math.min(Math.max(1, page), MAX_PAGES);

  const logs = await db.logs.findMany({
    skip: (safePage - 1) * PAGE_SIZE,
    take: PAGE_SIZE,
  });

  return { logs, page: safePage, pageSize: PAGE_SIZE };
});

// GOOD: Bounded broadcast with limit
const MAX_BROADCAST_TARGETS = 20;

function broadcastToAll(channel: string, data: unknown) {
  const windows = BrowserWindow.getAllWindows().slice(0, MAX_BROADCAST_TARGETS);

  console.assert(
    BrowserWindow.getAllWindows().length <= MAX_BROADCAST_TARGETS,
    `Window count exceeds maximum: ${BrowserWindow.getAllWindows().length}`
  );

  for (const win of windows) {
    if (!win.isDestroyed()) {
      win.webContents.send(channel, data);
    }
  }
}
```

**Guidelines:**
- All retry loops must have a `MAX_RETRIES` constant
- Paginate data passed over IPC — never send unbounded arrays
- Use `.slice(0, MAX)` before iterating collections of unknown size
- Define bounds as module-level constants
- Assert collection sizes at boundaries

---

### Rule 3: Controlled Memory — Controlled BrowserWindow Creation, Process Lifecycle

**Original Intent:** Prevent unbounded memory growth and allocation failures.

**Electron Adaptation:**

```typescript
// BAD: Creating windows on demand without tracking or limits
function openPreview(filePath: string) {
  const win = new BrowserWindow({ width: 800, height: 600 });
  win.loadFile(filePath);
  // No reference kept, no cleanup, no limit — memory leak
}

// BAD: Sending entire datasets over IPC
ipcMain.handle('export-data', async () => {
  const allRecords = await db.records.findMany();  // Could be 500MB
  return allRecords;  // Serialized across process boundary
});

// BAD: Never cleaning up webContents
function createTab(url: string) {
  const view = new WebContentsView();
  mainWindow.contentView.addChildView(view);
  view.webContents.loadURL(url);
  // View is never removed or destroyed
}

// GOOD: Window registry with upper bound and cleanup
const MAX_WINDOWS = 10;
const windowRegistry = new Map<string, BrowserWindow>();

function createWindow(id: string, options: BrowserWindowConstructorOptions): BrowserWindow | null {
  if (windowRegistry.size >= MAX_WINDOWS) {
    dialog.showErrorBox('Limit reached', `Cannot open more than ${MAX_WINDOWS} windows.`);
    return null;
  }

  const win = new BrowserWindow({
    ...options,
    webPreferences: {
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  windowRegistry.set(id, win);

  win.on('closed', () => {
    windowRegistry.delete(id);
  });

  return win;
}

// GOOD: Stream large data via file instead of IPC
const MAX_IPC_PAYLOAD = 5 * 1024 * 1024;  // 5MB

ipcMain.handle('export-data', async (event, rawParams: unknown) => {
  validateSender(event);
  const params = ExportSchema.parse(rawParams);

  const tmpPath = path.join(app.getPath('temp'), `export-${Date.now()}.json`);
  const records = db.records.stream(params.query);

  const writeStream = createWriteStream(tmpPath);
  for await (const chunk of records) {
    writeStream.write(JSON.stringify(chunk) + '\n');
  }
  writeStream.end();

  return { filePath: tmpPath };  // Send path, not data
});

// GOOD: Proper view cleanup
const viewRegistry = new Map<string, WebContentsView>();

function removeTab(tabId: string) {
  const view = viewRegistry.get(tabId);
  if (view) {
    mainWindow.contentView.removeChildView(view);
    view.webContents.close();
    viewRegistry.delete(tabId);
  }
}

app.on('before-quit', () => {
  for (const [id] of viewRegistry) {
    removeTab(id);
  }
});
```

**Guidelines:**
- Track all BrowserWindows in a registry with an upper bound
- Clean up webContents on window close events
- Never send datasets larger than 5MB over IPC — use file paths or streaming
- Use `app.on('before-quit')` and `win.on('closed')` for cleanup
- Monitor process memory with `process.memoryUsage()` in development

---

### Rule 4: Short Functions — Split Main, Preload, and Renderer (60 Lines)

**Original Intent:** Ensure functions are small enough to understand, test, and verify.

**Electron Adaptation:**

```typescript
// BAD: Monolithic main.ts with everything in one file
// main.ts — 500+ lines
import { app, BrowserWindow, ipcMain, Menu, Tray, dialog, shell } from 'electron';

app.whenReady().then(() => {
  const win = new BrowserWindow({ /* ... */ });
  win.loadFile('index.html');

  // 50 lines of menu setup...
  // 30 lines of tray setup...
  // 200 lines of IPC handlers for files, settings, auth, export...
  // 50 lines of auto-updater...
  // 40 lines of window management...
});

// BAD: Giant preload exposing dozens of methods
contextBridge.exposeInMainWorld('api', {
  readFile: (p: string) => ipcRenderer.invoke('file:read', p),
  writeFile: (p: string, c: string) => ipcRenderer.invoke('file:write', { p, c }),
  deleteFile: (p: string) => ipcRenderer.invoke('file:delete', p),
  listFiles: (dir: string) => ipcRenderer.invoke('file:list', dir),
  getSettings: () => ipcRenderer.invoke('settings:get'),
  updateSetting: (k: string, v: unknown) => ipcRenderer.invoke('settings:update', { k, v }),
  login: (u: string, p: string) => ipcRenderer.invoke('auth:login', { u, p }),
  logout: () => ipcRenderer.invoke('auth:logout'),
  exportData: (q: unknown) => ipcRenderer.invoke('export:run', q),
  // ... 20 more methods
});

// GOOD: Structured main process — separate files by domain
// main.ts (~30 lines)
import { app, BrowserWindow } from 'electron';
import { createMainWindow } from './windows/main-window';
import { registerFileHandlers } from './ipc/file-handlers';
import { registerSettingsHandlers } from './ipc/settings-handlers';
import { setupMenu } from './menu';

app.whenReady().then(() => {
  const mainWindow = createMainWindow();
  registerFileHandlers();
  registerSettingsHandlers();
  setupMenu(mainWindow);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// ipc/file-handlers.ts (~40 lines)
import { ipcMain } from 'electron';
import { ReadFileSchema, WriteFileSchema } from '../schemas/file-schemas';
import { validateSender } from '../security/validate-sender';

export function registerFileHandlers() {
  ipcMain.handle('file:read', async (event, rawData: unknown) => {
    validateSender(event);
    const { filePath } = ReadFileSchema.parse(rawData);
    const content = await fs.readFile(filePath, 'utf-8');
    return { content };
  });

  ipcMain.handle('file:write', async (event, rawData: unknown) => {
    validateSender(event);
    const { filePath, content } = WriteFileSchema.parse(rawData);
    await fs.writeFile(filePath, content);
    return { success: true };
  });
}

// GOOD: Focused preload — minimal bridge surface
// preload.ts (~15 lines)
import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('fileApi', {
  read: (params: { path: string }) => ipcRenderer.invoke('file:read', params),
  write: (params: { path: string; content: string }) => ipcRenderer.invoke('file:write', params),
});

contextBridge.exposeInMainWorld('settingsApi', {
  get: () => ipcRenderer.invoke('settings:get'),
  update: (params: { key: string; value: unknown }) => ipcRenderer.invoke('settings:update', params),
});
```

**Guidelines:**
- Maximum 60 lines per function or handler
- Separate main process into: lifecycle, IPC handlers (by domain), window management, menu
- Keep preload scripts minimal — just expose the bridge
- One IPC handler domain per file
- Renderer code follows standard frontend component rules

---

### Rule 5: Validation — Validate ALL IPC Messages with Schemas

**Original Intent:** Defensive programming catches bugs early; assertions document invariants.

**Electron Adaptation:**

```typescript
import { z } from 'zod';

// BAD: Trusting IPC data without validation
ipcMain.handle('save-file', async (event, filePath: string, content: string) => {
  await fs.writeFile(filePath, content);  // Attacker-controlled path and content!
  return { success: true };
});

// BAD: No sender validation
ipcMain.handle('delete-file', async (event, filePath: string) => {
  await fs.unlink(filePath);  // Any renderer — even injected code — can delete files
});

// BAD: No permission check
ipcMain.handle('run-command', async (event, cmd: string) => {
  return execSync(cmd).toString();  // Remote code execution!
});

// GOOD: Schema validation for every IPC channel
const SaveFileSchema = z.object({
  filePath: z.string()
    .max(260)
    .refine(p => !p.includes('..'), 'Path traversal detected')
    .refine(p => p.startsWith('/allowed/dir/'), 'Path outside allowed directory'),
  content: z.string().max(10 * 1024 * 1024),  // 10MB max
});

ipcMain.handle('save-file', async (event, rawData: unknown) => {
  validateSender(event);
  const result = SaveFileSchema.safeParse(rawData);

  if (!result.success) {
    return { success: false, error: result.error.flatten() };
  }

  await fs.writeFile(result.data.filePath, result.data.content);
  return { success: true };
});

// GOOD: Sender validation helper
const EXPECTED_ORIGINS = new Set(['app://myapp', 'https://myapp.local']);

function validateSender(event: Electron.IpcMainInvokeEvent): void {
  const url = new URL(event.senderFrame.url);

  console.assert(
    EXPECTED_ORIGINS.has(url.origin),
    `Untrusted IPC sender: ${url.origin}`
  );

  if (!EXPECTED_ORIGINS.has(url.origin)) {
    throw new Error(`Untrusted sender: ${url.origin}`);
  }
}

// GOOD: Reusable secure handler factory
function createSecureHandler<T>(
  schema: z.ZodSchema<T>,
  handler: (event: Electron.IpcMainInvokeEvent, data: T) => Promise<unknown>,
) {
  return async (event: Electron.IpcMainInvokeEvent, rawData: unknown) => {
    validateSender(event);

    const result = schema.safeParse(rawData);
    if (!result.success) {
      return { success: false, error: 'Validation failed' };
    }

    return handler(event, result.data);
  };
}

// Usage
ipcMain.handle('file:read', createSecureHandler(ReadFileSchema, async (event, data) => {
  const content = await fs.readFile(data.filePath, 'utf-8');
  return { success: true, content };
}));
```

**Guidelines:**
- Validate EVERY IPC message with Zod or a similar schema library
- Validate sender identity via `event.senderFrame.url` against an allowlist
- Sanitize file paths — reject path traversal (`..`) and paths outside allowed directories
- Use `safeParse` and return structured errors instead of crashing
- Create a reusable `createSecureHandler` factory to enforce validation consistently
- Never expose shell execution or unrestricted filesystem access over IPC

---

### Rule 6: Minimal Scope — Context Isolation, Minimal contextBridge Surface

**Original Intent:** Reduce state complexity and potential for misuse.

**Electron Adaptation:**

```typescript
// BAD: Disabling all security defaults
const win = new BrowserWindow({
  webPreferences: {
    contextIsolation: false,   // Exposes preload globals to page
    nodeIntegration: true,     // Full Node.js in renderer
    sandbox: false,            // No OS-level sandboxing
  },
});

// BAD: Exposing raw Node.js modules through contextBridge
contextBridge.exposeInMainWorld('node', {
  fs: require('fs'),
  childProcess: require('child_process'),
  path: require('path'),
});

// BAD: Single giant API surface
contextBridge.exposeInMainWorld('electron', {
  send: (channel: string, ...args: unknown[]) => ipcRenderer.send(channel, ...args),
  invoke: (channel: string, ...args: unknown[]) => ipcRenderer.invoke(channel, ...args),
  on: (channel: string, callback: Function) => ipcRenderer.on(channel, callback),
  // Exposes ALL channels — no restrictions
});

// GOOD: Secure BrowserWindow with all defaults enforced
const win = new BrowserWindow({
  webPreferences: {
    contextIsolation: true,     // Default since Electron 12
    sandbox: true,              // Default since Electron 20
    nodeIntegration: false,     // MUST remain false
    preload: path.join(__dirname, 'preload.js'),
  },
});

// GOOD: Narrow, purpose-specific API surface
// preload-settings.ts — only for the settings window
contextBridge.exposeInMainWorld('settingsApi', {
  getSettings: () => ipcRenderer.invoke('settings:get'),
  updateSetting: (key: string, value: unknown) =>
    ipcRenderer.invoke('settings:update', { key, value }),
});

// preload-editor.ts — only for the editor window
contextBridge.exposeInMainWorld('editorApi', {
  openFile: (filePath: string) => ipcRenderer.invoke('file:read', { filePath }),
  saveFile: (filePath: string, content: string) =>
    ipcRenderer.invoke('file:write', { filePath, content }),
});

// GOOD: Namespace IPC channels by domain
// Channels follow 'domain:action' convention
ipcMain.handle('file:read', fileReadHandler);
ipcMain.handle('file:write', fileWriteHandler);
ipcMain.handle('settings:get', settingsGetHandler);
ipcMain.handle('settings:update', settingsUpdateHandler);
ipcMain.handle('auth:login', authLoginHandler);
ipcMain.handle('auth:logout', authLogoutHandler);

// GOOD: One-way listener with stripped event
contextBridge.exposeInMainWorld('updates', {
  onProgress: (callback: (percent: number) => void) => {
    // Strip the IpcRendererEvent — never expose it to the page
    ipcRenderer.on('update:progress', (_event, percent: number) => callback(percent));
  },
});
```

**Guidelines:**
- Never disable `contextIsolation` or enable `nodeIntegration`
- Enable `sandbox: true` (default since Electron 20)
- Expose the smallest possible API surface through `contextBridge`
- Use separate preload scripts for different window types
- Namespace IPC channels by domain (e.g., `'file:read'`, `'settings:get'`)
- Never expose raw Node.js modules or `ipcRenderer.on`/`send` directly
- Strip the `IpcRendererEvent` object when forwarding events to the renderer

---

### Rule 7: Check Return Values — Handle All IPC Responses, App Lifecycle, Window Events

**Original Intent:** Never ignore errors; verify inputs at trust boundaries.

**Electron Adaptation:**

```typescript
// BAD: Ignoring app lifecycle
const win = new BrowserWindow({ /* ... */ });
win.loadFile('index.html');
// App may not be ready yet — crash on some platforms

// BAD: Fire-and-forget IPC
// renderer.ts
window.api.saveFile(path, content);  // No error handling, no response check

// BAD: Ignoring window events
function createWindow() {
  const win = new BrowserWindow({ /* ... */ });
  win.loadFile('index.html');
  // No 'closed' handler — memory leak
  // No 'unresponsive' handler — user stuck
  // No 'render-process-gone' handler — blank screen
}

// GOOD: Proper app lifecycle handling
app.whenReady().then(() => {
  createMainWindow();
  registerIpcHandlers();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
});

app.on('web-contents-created', (_event, contents) => {
  contents.on('will-navigate', (event, url) => {
    if (new URL(url).origin !== EXPECTED_ORIGIN) {
      event.preventDefault();
    }
  });
});

// GOOD: Typed IPC result handling in renderer
type IpcResult<T> =
  | { success: true; data: T }
  | { success: false; error: string };

async function saveFile(filePath: string, content: string): Promise<void> {
  const result: IpcResult<void> = await window.fileApi.save({ filePath, content });

  if (!result.success) {
    showNotification('Save failed', result.error);
    return;
  }

  showNotification('File saved');
}

// GOOD: Comprehensive window event handling
function createMainWindow(): BrowserWindow {
  const win = new BrowserWindow({ /* ... */ });

  win.on('closed', () => {
    windowRegistry.delete('main');
  });

  win.on('unresponsive', () => {
    dialog.showMessageBox({
      message: 'The window is not responding.',
      buttons: ['Wait', 'Reload', 'Close'],
    }).then(({ response }) => {
      if (response === 1) win.reload();
      if (response === 2) win.close();
    });
  });

  win.webContents.on('render-process-gone', (_event, details) => {
    logger.error('Renderer process gone', details);

    if (details.reason !== 'clean-exit') {
      dialog.showMessageBox({
        message: 'The window has crashed.',
        buttons: ['Reload', 'Close'],
      }).then(({ response }) => {
        if (response === 0) win.reload();
        else win.close();
      });
    }
  });

  win.loadFile('index.html');
  return win;
}

// GOOD: Handle IPC errors in main process
ipcMain.handle('file:read', async (event, rawData: unknown) => {
  try {
    validateSender(event);
    const { filePath } = ReadFileSchema.parse(rawData);
    const content = await fs.readFile(filePath, 'utf-8');
    return { success: true, data: { content } };
  } catch (error) {
    logger.error('file:read failed', error);
    return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
  }
});
```

**Guidelines:**
- Always await `app.whenReady()` before creating windows
- Use `ipcRenderer.invoke` (returns Promise) instead of `ipcRenderer.send` (fire-and-forget)
- Handle `'unresponsive'` and `'render-process-gone'` events on every window
- Use typed result wrappers (`IpcResult<T>`) for all IPC responses
- Handle `'closed'`, `'window-all-closed'`, `'activate'` lifecycle events
- Wrap all `ipcMain.handle` callbacks in try/catch with structured error returns

---

### Rule 8: Limit Metaprogramming — No eval, No nodeIntegration, Strict CSP

**Original Intent:** Avoid constructs that create unmaintainable, unanalyzable code.

**Electron Adaptation:**

```typescript
// BAD: Using eval or dynamic code execution
function runUserScript(code: string) {
  eval(code);  // Remote code execution in a privileged context
}

// BAD: Using the deprecated remote module
import { remote } from '@electron/remote';
const win = remote.getCurrentWindow();  // Bypasses IPC security boundary

// BAD: No Content Security Policy
const win = new BrowserWindow({ /* ... */ });
win.loadFile('index.html');
// Any injected script can execute freely

// BAD: Opening external URLs without validation
ipcMain.handle('open-link', async (event, url: string) => {
  await shell.openExternal(url);  // Could open malicious protocols
});

// GOOD: Strict Content Security Policy
import { session } from 'electron';

session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
  callback({
    responseHeaders: {
      ...details.responseHeaders,
      'Content-Security-Policy': [
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'",
      ],
    },
  });
});

// GOOD: Restrict navigation to trusted origins
function configureSecurityHandlers(win: BrowserWindow) {
  win.webContents.on('will-navigate', (event, url) => {
    const parsed = new URL(url);
    if (parsed.origin !== EXPECTED_ORIGIN) {
      event.preventDefault();
      logger.warn('Blocked navigation to', url);
    }
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    if (isTrustedExternalUrl(url)) {
      shell.openExternal(url);
    }
    return { action: 'deny' };  // Always deny new window creation
  });
}

// GOOD: Validated external URL opening
const ALLOWED_PROTOCOLS = new Set(['https:']);
const ALLOWED_HOSTS = new Set(['docs.example.com', 'support.example.com', 'github.com']);

function isTrustedExternalUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return ALLOWED_PROTOCOLS.has(parsed.protocol) && ALLOWED_HOSTS.has(parsed.hostname);
  } catch {
    return false;
  }
}

ipcMain.handle('open-external', async (event, rawUrl: unknown) => {
  validateSender(event);
  const url = z.string().url().parse(rawUrl);

  if (!isTrustedExternalUrl(url)) {
    return { success: false, error: 'URL not in allowlist' };
  }

  await shell.openExternal(url);
  return { success: true };
});

// GOOD: Enforce security on webview creation
app.on('web-contents-created', (_event, contents) => {
  contents.on('will-attach-webview', (event, webPreferences, params) => {
    // Strip preload scripts from webviews
    delete webPreferences.preload;

    // Enforce security defaults
    webPreferences.nodeIntegration = false;
    webPreferences.contextIsolation = true;

    // Block loading untrusted origins
    if (!params.src.startsWith('https://trusted.example.com')) {
      event.preventDefault();
    }
  });
});
```

**Guidelines:**
- Never use `eval()`, `new Function()`, or enable `nodeIntegration: true`
- Never use the deprecated `@electron/remote` module
- Define a strict Content Security Policy via `session.webRequest.onHeadersReceived`
- Whitelist URLs for `shell.openExternal` — never pass untrusted URLs directly
- Restrict navigation with `will-navigate` and `setWindowOpenHandler`
- Intercept `will-attach-webview` to enforce security on embedded webviews
- Use custom protocols (`app://`) instead of `file://` for loading app content

---

### Rule 9: Type Safety — Strict TypeScript, Typed IPC Channels

**Original Intent:** (C: Restrict pointer usage for safety)

**Electron Adaptation:**

```typescript
// BAD: Untyped IPC handlers
ipcMain.handle('do-thing', async (event, data: any) => {
  return data.value.nested.prop;  // No type safety, crash at runtime
});

// BAD: Untyped contextBridge
contextBridge.exposeInMainWorld('api', {
  doAnything: (...args: any[]) => ipcRenderer.invoke('do-thing', ...args),
});

// BAD: Accessing window.api without types
const result = (window as any).api.doAnything('hello');

// GOOD: Shared IPC channel type definitions
// shared/ipc-channels.ts
export interface IpcChannelMap {
  'file:read': {
    params: { filePath: string };
    result: { content: string };
  };
  'file:write': {
    params: { filePath: string; content: string };
    result: { success: boolean };
  };
  'settings:get': {
    params: void;
    result: Settings;
  };
  'settings:update': {
    params: { key: keyof Settings; value: Settings[keyof Settings] };
    result: Settings;
  };
}

// GOOD: Type-safe invoke wrapper in preload
// preload.ts
function createInvoker<K extends keyof IpcChannelMap>(channel: K) {
  return (
    ...args: IpcChannelMap[K]['params'] extends void ? [] : [IpcChannelMap[K]['params']]
  ): Promise<IpcChannelMap[K]['result']> =>
    ipcRenderer.invoke(channel, ...args);
}

contextBridge.exposeInMainWorld('api', {
  readFile: createInvoker('file:read'),
  writeFile: createInvoker('file:write'),
  getSettings: createInvoker('settings:get'),
  updateSettings: createInvoker('settings:update'),
});

// GOOD: Renderer-side type declarations
// renderer/global.d.ts
import type { IpcChannelMap } from '../shared/ipc-channels';

interface ElectronApi {
  readFile: (params: IpcChannelMap['file:read']['params']) =>
    Promise<IpcChannelMap['file:read']['result']>;
  writeFile: (params: IpcChannelMap['file:write']['params']) =>
    Promise<IpcChannelMap['file:write']['result']>;
  getSettings: () => Promise<IpcChannelMap['settings:get']['result']>;
  updateSettings: (params: IpcChannelMap['settings:update']['params']) =>
    Promise<IpcChannelMap['settings:update']['result']>;
}

declare global {
  interface Window {
    api: ElectronApi;
  }
}

// GOOD: Branded types for sensitive identifiers
type FilePath = string & { readonly __brand: 'FilePath' };
type WindowId = string & { readonly __brand: 'WindowId' };

function toFilePath(raw: string): FilePath {
  if (raw.includes('..')) throw new Error('Path traversal');
  return raw as FilePath;
}

function getWindow(id: WindowId): BrowserWindow | undefined {
  return windowRegistry.get(id);
  // Cannot accidentally pass a FilePath here
}
```

**tsconfig.json:**
```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "noUncheckedIndexedAccess": true
  }
}
```

**Guidelines:**
- Enable all strict TypeScript options
- Define a shared `IpcChannelMap` type for all IPC channels
- Type-safe `contextBridge` APIs — no `any` in exposed functions
- Declare `window.api` types in a `global.d.ts` for renderer code
- Never use `as any` to bypass IPC types
- Use branded types for sensitive identifiers (file paths, window IDs)

---

### Rule 10: Static Analysis — ESLint, @electron/fuses, Electronegativity

**Original Intent:** Catch issues at development time; use every available tool.

**Electron Adaptation:**

```typescript
// eslint.config.mjs
import tseslint from 'typescript-eslint';

export default tseslint.config(
  ...tseslint.configs.strictTypeChecked,
  {
    rules: {
      // Security
      'no-eval': 'error',
      'no-implied-eval': 'error',
      'no-new-func': 'error',

      // TypeScript strict
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-non-null-assertion': 'error',
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/strict-boolean-expressions': 'error',
      '@typescript-eslint/no-unsafe-argument': 'error',
    },
  },
);
```

```typescript
// build/configure-fuses.ts — run after packaging
import { flipFuses, FuseVersion, FuseV1Options } from '@electron/fuses';

async function configureFuses(electronPath: string) {
  await flipFuses(electronPath, {
    version: FuseVersion.V1,
    [FuseV1Options.RunAsNode]: false,                          // Prevent ELECTRON_RUN_AS_NODE
    [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false, // Prevent NODE_OPTIONS injection
    [FuseV1Options.EnableNodeCliInspectArguments]: false,        // Prevent --inspect debugging
    [FuseV1Options.OnlyLoadAppFromAsar]: true,                  // Prevent code replacement
  });
}
```

```bash
# Required CI pipeline
tsc --noEmit                          # Type checking
eslint --max-warnings 0 src/          # Zero warnings
npx electronegativity -i .            # Electron security linting
npm audit                             # Dependency vulnerabilities
```

```typescript
// Runtime security audit — enforce webPreferences on all webContents
app.on('web-contents-created', (_event, contents) => {
  // Block new window creation by default
  contents.setWindowOpenHandler(() => ({ action: 'deny' }));

  // Enforce webview security
  contents.on('will-attach-webview', (event, webPreferences) => {
    delete webPreferences.preload;
    webPreferences.nodeIntegration = false;
    webPreferences.contextIsolation = true;
  });

  // Block navigation to untrusted origins
  contents.on('will-navigate', (event, url) => {
    if (new URL(url).origin !== EXPECTED_ORIGIN) {
      event.preventDefault();
    }
  });
});
```

**Guidelines:**
- Run ESLint with strict TypeScript rules, zero warnings policy
- Use `@electron/fuses` to disable dangerous runtime features after packaging
- Run `electronegativity` for Electron-specific security linting
- Intercept `web-contents-created` to enforce security defaults on all webContents
- Intercept `will-attach-webview` to enforce security on embedded webviews
- Run `npm audit` regularly, use Dependabot or Renovate for dependency updates
- No `// @ts-ignore` or `eslint-disable` without documented justification

---

## Summary: Electron Adaptation

| # | Original Rule | Electron Guideline |
|---|---------------|--------------------|
| 1 | No goto/recursion | Simple IPC patterns, no recursive message chains, guard clauses |
| 2 | Fixed loop bounds | Bounded retries, paginate IPC data, `.slice()` before iteration |
| 3 | No dynamic allocation | Controlled BrowserWindow creation, process lifecycle cleanup |
| 4 | 60-line functions | Split main/preload/renderer into focused modules by domain |
| 5 | 2+ assertions/function | Validate all IPC with Zod, validate sender identity |
| 6 | Minimize scope | Context isolation, minimal contextBridge surface, sandbox enabled |
| 7 | Check returns | Handle IPC responses, app lifecycle, window crash events |
| 8 | Limit preprocessor | No eval/nodeIntegration, strict CSP, no remote module |
| 9 | Restrict pointers | Strict TypeScript, typed IPC channels, no `any` in contextBridge |
| 10 | All warnings enabled | ESLint strict, @electron/fuses, electronegativity, npm audit |

---

## References

- [Original Power of 10 Paper](https://spinroot.com/gerard/pdf/P10.pdf) — Gerard Holzmann
- [Electron Security Tutorial](https://www.electronjs.org/docs/latest/tutorial/security) — Official Security Checklist
- [Electron Process Model](https://www.electronjs.org/docs/latest/tutorial/process-model) — Main, Renderer, Preload
- [Electron Context Isolation](https://www.electronjs.org/docs/latest/tutorial/context-isolation) — contextBridge API
- [Electron IPC Guide](https://www.electronjs.org/docs/latest/tutorial/ipc) — Inter-Process Communication
- [Electron Performance](https://www.electronjs.org/docs/latest/tutorial/performance) — Performance Best Practices
- [@electron/fuses](https://www.npmjs.com/package/@electron/fuses) — Runtime Feature Toggles
- [Electronegativity](https://github.com/nicedoc/electronegativity) — Electron Security Linter
- [Zod Documentation](https://zod.dev/) — Schema Validation
