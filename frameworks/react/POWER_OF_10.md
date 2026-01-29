# The Power of 10 Rules for Safety-Critical React Code

## Background

The **Power of 10 Rules** were created in 2006 by **Gerard J. Holzmann** at NASA's Jet Propulsion Laboratory (JPL) Laboratory for Reliable Software. These rules were designed for writing safety-critical code in C that could be effectively analyzed by static analysis tools.

The rules were incorporated into JPL's institutional coding standard and used for major missions including the **Mars Science Laboratory** (Curiosity Rover, 2012).

> *"If these rules seem draconian at first, bear in mind that they are meant to make it possible to check safety-critical code where human lives can very literally depend on its correctness."* — Gerard Holzmann

**Note:** This adaptation focuses on client-side React patterns (hooks, state, effects) and works with any React setup: Vite, Create React App, Remix, Astro, etc. For Next.js-specific patterns (Server Components, Server Actions), see the NextJS Power of 10.

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

## The Power of 10 Rules — React Edition

### Rule 1: Simple Control Flow — No Recursive Components, Guard Clauses

**Original Intent:** Eliminate complex control flow that impedes static analysis and can cause stack overflows.

**React Adaptation:**

```tsx
// BAD: Recursive component (stack overflow risk)
function TreeNode({ node }: { node: TreeNode }) {
  return (
    <div>
      {node.label}
      {node.children.map(child => (
        <TreeNode key={child.id} node={child} />  // Unbounded recursion
      ))}
    </div>
  );
}

// BAD: Deeply nested conditionals
function UserProfile({ user, isActive, canView }: Props) {
  if (user) {
    if (isActive) {
      if (canView) {
        return <Profile user={user} />;
      }
    }
  }
  return null;
}

// GOOD: Iterative rendering with bounded depth
const MAX_DEPTH = 5;

function TreeView({ nodes }: { nodes: TreeNode[] }) {
  const renderLevel = (items: TreeNode[], depth: number): React.ReactNode => {
    if (depth >= MAX_DEPTH) return null;

    return items.map(node => (
      <div key={node.id} style={{ marginLeft: depth * 16 }}>
        {node.label}
        {node.children && renderLevel(node.children, depth + 1)}
      </div>
    ));
  };

  return <div>{renderLevel(nodes, 0)}</div>;
}

// GOOD: Guard clauses with early returns
function UserProfile({ user, isActive, canView }: Props) {
  if (!user) return null;
  if (!isActive) return <InactiveMessage />;
  if (!canView) return <AccessDenied />;

  return <Profile user={user} />;
}
```

**Guidelines:**
- No recursive components — use iterative rendering with explicit depth limits
- Use guard clauses for early returns (reduces nesting)
- Maximum 3-4 levels of JSX nesting
- Flatten component trees by extracting sub-components
- Avoid ternary chains — use early returns or lookup objects

---

### Rule 2: Bounded Loops — Paginate and Virtualize Lists

**Original Intent:** Ensure all loops terminate and can be analyzed statically.

**React Adaptation:**

```tsx
// BAD: Rendering unbounded list
function UserList({ users }: { users: User[] }) {
  return (
    <ul>
      {users.map(user => (  // Could be 100,000 items
        <UserCard key={user.id} user={user} />
      ))}
    </ul>
  );
}

// BAD: Unbounded data fetching
const [allUsers, setAllUsers] = useState<User[]>([]);

useEffect(() => {
  fetchAllUsers().then(setAllUsers);  // Loads everything into memory
}, []);

// GOOD: Bounded list with explicit limit
const MAX_VISIBLE_ITEMS = 100;

function UserList({ users }: { users: User[] }) {
  const boundedUsers = users.slice(0, MAX_VISIBLE_ITEMS);

  console.assert(
    boundedUsers.length <= MAX_VISIBLE_ITEMS,
    `Rendered ${users.length} items, expected max ${MAX_VISIBLE_ITEMS}`
  );

  return (
    <ul>
      {boundedUsers.map(user => (
        <UserCard key={user.id} user={user} />
      ))}
      {users.length > MAX_VISIBLE_ITEMS && (
        <li>{users.length - MAX_VISIBLE_ITEMS} more items...</li>
      )}
    </ul>
  );
}

// GOOD: Virtualized list for large datasets
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualizedList({ items }: { items: Item[] }) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50,
    overscan: 5,
  });

  return (
    <div ref={parentRef} style={{ height: 400, overflow: 'auto' }}>
      <div style={{ height: virtualizer.getTotalSize() }}>
        {virtualizer.getVirtualItems().map(virtualRow => (
          <div key={virtualRow.key} style={{ height: virtualRow.size }}>
            <ItemCard item={items[virtualRow.index]} />
          </div>
        ))}
      </div>
    </div>
  );
}

// GOOD: Paginated data fetching
const PAGE_SIZE = 20;

function usePaginatedUsers() {
  const [page, setPage] = useState(1);
  const [users, setUsers] = useState<User[]>([]);

  useEffect(() => {
    const controller = new AbortController();

    fetchUsers({ page, limit: PAGE_SIZE }, controller.signal)
      .then(setUsers)
      .catch(err => {
        if (err.name !== 'AbortError') console.error(err);
      });

    return () => controller.abort();
  }, [page]);

  return { users, page, setPage };
}
```

**Guidelines:**
- Always `.slice(0, MAX)` before mapping arrays of unknown size
- Use virtualization (@tanstack/react-virtual) for lists > 100 items
- Paginate API requests — never fetch unbounded data
- Define bounds as constants at module level
- Assert collection sizes in development

---

### Rule 3: Controlled Memory — Minimize State, Derive Values

**Original Intent:** Prevent unbounded memory growth and allocation failures.

**React Adaptation:**

```tsx
// BAD: Duplicating derived state
function ProductList({ products }: Props) {
  const [filteredProducts, setFilteredProducts] = useState(products);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    // Storing derived data = memory waste + sync bugs
    setFilteredProducts(
      products.filter(p => p.name.includes(searchTerm))
    );
  }, [products, searchTerm]);

  return <List items={filteredProducts} />;
}

// BAD: Storing fetched data in multiple places
function Dashboard() {
  const [users, setUsers] = useState<User[]>([]);
  const [posts, setPosts] = useState<Post[]>([]);
  const [userPosts, setUserPosts] = useState<Map<string, Post[]>>(new Map());

  // Now you have 3 sources of truth to keep in sync...
}

// GOOD: Derive values during render (no extra state)
function ProductList({ products }: Props) {
  const [searchTerm, setSearchTerm] = useState('');

  // Derived during render — no state sync needed
  const filteredProducts = products.filter(p =>
    p.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <>
      <SearchInput value={searchTerm} onChange={setSearchTerm} />
      <List items={filteredProducts} />
    </>
  );
}

// GOOD: useMemo for expensive derivations only
function ExpensiveList({ items }: { items: Item[] }) {
  const [filter, setFilter] = useState('');

  // Only memoize truly expensive computations
  const processed = useMemo(() => {
    console.assert(items.length < 10000, 'Consider pagination');
    return items
      .filter(item => item.name.includes(filter))
      .sort((a, b) => b.score - a.score)
      .slice(0, 100);
  }, [items, filter]);

  return <List items={processed} />;
}

// GOOD: Single source of truth with derived access
function useUserData(userId: string) {
  const [user, setUser] = useState<User | null>(null);

  // Derive — don't duplicate
  const isAdmin = user?.role === 'admin';
  const displayName = user ? `${user.firstName} ${user.lastName}` : '';

  return { user, isAdmin, displayName, setUser };
}
```

**Guidelines:**
- Minimize `useState` — ask "can I derive this?" first
- Never store derived data in state (causes sync bugs)
- Use `useMemo` only for genuinely expensive computations
- React 19's compiler auto-memoizes — manual memo is often unnecessary
- One source of truth: derive views, don't duplicate data

---

### Rule 4: Short Components — 60 Lines Maximum

**Original Intent:** Ensure functions are small enough to understand, test, and verify.

**React Adaptation:**

```tsx
// BAD: Monolithic component
function UserDashboard({ userId }: { userId: string }) {
  // 200+ lines: data fetching, state, handlers, complex JSX...
  const [user, setUser] = useState(null);
  const [posts, setPosts] = useState([]);
  const [isEditing, setIsEditing] = useState(false);
  // ... 15 more useState calls
  // ... 10 useEffect calls
  // ... 20 event handlers
  // ... 100 lines of JSX
}

// GOOD: Decomposed into focused components
function UserDashboard({ userId }: { userId: string }) {
  const { user, isLoading, error } = useUser(userId);

  if (isLoading) return <DashboardSkeleton />;
  if (error) return <ErrorMessage error={error} />;
  if (!user) return <NotFound />;

  return (
    <main>
      <UserHeader user={user} />
      <UserStats userId={userId} />
      <UserPosts userId={userId} />
      <UserActivity userId={userId} />
    </main>
  );
}

// GOOD: Extract custom hooks for reusable logic
function useUser(userId: string) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setIsLoading(true);

    fetchUser(userId, controller.signal)
      .then(setUser)
      .catch(err => {
        if (err.name !== 'AbortError') setError(err);
      })
      .finally(() => setIsLoading(false));

    return () => controller.abort();
  }, [userId]);

  return { user, isLoading, error };
}

// GOOD: Extract event handlers
function useFormHandlers(initialData: FormData) {
  const [data, setData] = useState(initialData);

  const handlers = {
    onChange: (field: keyof FormData, value: string) => {
      setData(prev => ({ ...prev, [field]: value }));
    },
    onReset: () => setData(initialData),
    onSubmit: async () => {
      await saveData(data);
    },
  };

  return { data, ...handlers };
}
```

**Guidelines:**
- Maximum 60 lines per component (including hooks and JSX)
- One component per file
- Extract custom hooks for stateful logic (≤30 lines each)
- Extract event handlers into separate functions or hooks
- Split large forms into step/section components

---

### Rule 5: Validation — TypeScript + Zod for Runtime Checks

**Original Intent:** Defensive programming catches bugs early; assertions document invariants.

**React Adaptation:**

```tsx
// BAD: Trusting external data
function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then(res => res.json())
      .then(setUser);  // No validation — trusting API blindly
  }, [userId]);

  return <div>{user?.email}</div>;  // Could crash if shape is wrong
}

// BAD: No prop validation
function PriceDisplay({ price, currency }) {  // No types!
  return <span>{currency}{price.toFixed(2)}</span>;  // Crashes if price is string
}

// GOOD: Zod schema for API responses
import { z } from 'zod';

const UserSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  name: z.string().min(1).max(100),
  role: z.enum(['admin', 'user', 'guest']),
  createdAt: z.string().datetime(),
});

type User = z.infer<typeof UserSchema>;

function useUser(userId: string) {
  const [user, setUser] = useState<User | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    fetch(`/api/users/${userId}`, { signal: controller.signal })
      .then(res => res.json())
      .then(data => {
        const result = UserSchema.safeParse(data);
        if (result.success) {
          setUser(result.data);
        } else {
          console.error('Invalid API response:', result.error);
          setError('Invalid user data received');
        }
      })
      .catch(err => {
        if (err.name !== 'AbortError') setError(err.message);
      });

    return () => controller.abort();
  }, [userId]);

  return { user, error };
}

// GOOD: Typed props with runtime assertions
interface PriceDisplayProps {
  price: number;
  currency: string;
  locale?: string;
}

function PriceDisplay({ price, currency, locale = 'en-US' }: PriceDisplayProps) {
  console.assert(typeof price === 'number', `price must be number, got ${typeof price}`);
  console.assert(price >= 0, `price must be non-negative, got ${price}`);

  return (
    <span>
      {new Intl.NumberFormat(locale, {
        style: 'currency',
        currency,
      }).format(price)}
    </span>
  );
}

// GOOD: Form validation with Zod
const ContactFormSchema = z.object({
  email: z.string().email('Invalid email address'),
  message: z.string().min(10, 'Message must be at least 10 characters'),
});

function ContactForm() {
  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const data = Object.fromEntries(formData);

    const result = ContactFormSchema.safeParse(data);
    if (!result.success) {
      setErrors(result.error.flatten().fieldErrors);
      return;
    }

    // Safe to use result.data
    submitContact(result.data);
  };

  return <form onSubmit={handleSubmit}>{/* ... */}</form>;
}
```

**Guidelines:**
- Validate ALL external data (API responses, URL params, localStorage)
- Use Zod/Valibot for runtime schema validation
- Use `safeParse` — don't throw on invalid data
- Infer TypeScript types from Zod schemas (single source of truth)
- Add `console.assert` for invariants in development

---

### Rule 6: Minimal Scope — Colocate State, Avoid Prop Drilling

**Original Intent:** Reduce state complexity and potential for misuse.

**React Adaptation:**

```tsx
// BAD: Global state for local concerns
// store.ts
const useGlobalStore = create((set) => ({
  isModalOpen: false,  // Only used in one component!
  modalData: null,
  setModalOpen: (open) => set({ isModalOpen: open }),
}));

// BAD: Prop drilling through many layers
function App() {
  const [user, setUser] = useState<User | null>(null);
  return <Layout user={user} setUser={setUser} />;
}
function Layout({ user, setUser }) {
  return <Sidebar user={user} setUser={setUser} />;
}
function Sidebar({ user, setUser }) {
  return <UserMenu user={user} setUser={setUser} />;
}
function UserMenu({ user, setUser }) {
  // Finally used here, 4 levels deep
}

// GOOD: Colocate state where it's used
function Modal() {
  const [isOpen, setIsOpen] = useState(false);  // Local to Modal

  return (
    <>
      <button onClick={() => setIsOpen(true)}>Open</button>
      {isOpen && (
        <Dialog onClose={() => setIsOpen(false)}>
          <ModalContent />
        </Dialog>
      )}
    </>
  );
}

// GOOD: Context for truly shared state (used sparingly)
const UserContext = createContext<User | null>(null);

function UserProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  // Fetch user once at top level
  useEffect(() => {
    fetchCurrentUser().then(setUser);
  }, []);

  return (
    <UserContext.Provider value={user}>
      {children}
    </UserContext.Provider>
  );
}

function useUser() {
  const user = useContext(UserContext);
  if (user === undefined) {
    throw new Error('useUser must be used within UserProvider');
  }
  return user;
}

// GOOD: Composition over prop drilling
function App() {
  const [user, setUser] = useState<User | null>(null);

  return (
    <Layout>
      <Sidebar>
        <UserMenu user={user} setUser={setUser} />
      </Sidebar>
    </Layout>
  );
}

function Layout({ children }: { children: React.ReactNode }) {
  return <div className="layout">{children}</div>;
}
```

**Guidelines:**
- Colocate state with the component that uses it
- Lift state only when multiple components need it
- Use Context sparingly — only for truly global data (user, theme, locale)
- Prefer composition (children) over prop drilling
- Don't put UI state (modals, tabs) in global stores

---

### Rule 7: Check Returns — Handle Loading, Error, and Empty States

**Original Intent:** Never ignore errors; verify inputs at trust boundaries.

**React Adaptation:**

```tsx
// BAD: Ignoring loading and error states
function UserProfile({ userId }: { userId: string }) {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    fetchUser(userId).then(setUser);
  }, [userId]);

  return <div>{user.name}</div>;  // Crashes when user is null!
}

// BAD: Swallowing errors
useEffect(() => {
  fetchData().catch(() => {});  // Silent failure
}, []);

// GOOD: Handle all states explicitly
type AsyncState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'success'; data: T };

function UserProfile({ userId }: { userId: string }) {
  const [state, setState] = useState<AsyncState<User>>({ status: 'idle' });

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: 'loading' });

    fetchUser(userId, controller.signal)
      .then(data => setState({ status: 'success', data }))
      .catch(error => {
        if (error.name !== 'AbortError') {
          setState({ status: 'error', error });
        }
      });

    return () => controller.abort();
  }, [userId]);

  switch (state.status) {
    case 'idle':
    case 'loading':
      return <ProfileSkeleton />;
    case 'error':
      return <ErrorMessage error={state.error} onRetry={() => {/* ... */}} />;
    case 'success':
      return <ProfileContent user={state.data} />;
  }
}

// GOOD: Custom hook with proper state handling
function useAsync<T>(asyncFn: () => Promise<T>, deps: unknown[]) {
  const [state, setState] = useState<AsyncState<T>>({ status: 'idle' });

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: 'loading' });

    asyncFn()
      .then(data => setState({ status: 'success', data }))
      .catch(error => {
        if (error.name !== 'AbortError') {
          setState({ status: 'error', error });
        }
      });

    return () => controller.abort();
  }, deps);

  return state;
}

// GOOD: Handle empty states
function UserList({ users }: { users: User[] }) {
  if (users.length === 0) {
    return <EmptyState message="No users found" />;
  }

  return (
    <ul>
      {users.map(user => (
        <UserCard key={user.id} user={user} />
      ))}
    </ul>
  );
}

// GOOD: Null checks before accessing properties
function UserAvatar({ user }: { user: User | null }) {
  if (!user) return <DefaultAvatar />;
  if (!user.avatarUrl) return <InitialsAvatar name={user.name} />;

  return <img src={user.avatarUrl} alt={user.name} />;
}
```

**Guidelines:**
- Always handle loading, error, success, and empty states
- Use discriminated unions for async state (not separate booleans)
- Never use non-null assertion (`!`) — handle null explicitly
- Log errors before displaying user-friendly messages
- Provide retry mechanisms for recoverable errors

---

### Rule 8: Limit Metaprogramming — No dangerouslySetInnerHTML, Explicit Code

**Original Intent:** Avoid constructs that create unmaintainable, unanalyzable code.

**React Adaptation:**

```tsx
// BAD: XSS vulnerability
function Comment({ html }: { html: string }) {
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

// BAD: Dynamic code execution
function DynamicRenderer({ code }: { code: string }) {
  useEffect(() => {
    eval(code);  // Never!
  }, [code]);
}

// BAD: Excessive HOC stacking
const EnhancedComponent = withAuth(
  withLogging(
    withErrorBoundary(
      withTheme(
        withI18n(BaseComponent)
      )
    )
  )
);

// BAD: Magic props spreading
function Button(props: any) {
  return <button {...props} />;  // What props are valid?
}

// GOOD: Sanitize if HTML is absolutely required
import DOMPurify from 'dompurify';

const ALLOWED_TAGS = ['b', 'i', 'em', 'strong', 'a', 'p', 'br'];
const ALLOWED_ATTR = ['href', 'target', 'rel'];

function SafeHTML({ html }: { html: string }) {
  const sanitized = DOMPurify.sanitize(html, {
    ALLOWED_TAGS,
    ALLOWED_ATTR,
  });

  return <div dangerouslySetInnerHTML={{ __html: sanitized }} />;
}

// GOOD: Use React components instead of HTML strings
function FormattedText({ text, format }: { text: string; format: 'bold' | 'italic' | 'normal' }) {
  switch (format) {
    case 'bold': return <strong>{text}</strong>;
    case 'italic': return <em>{text}</em>;
    default: return <span>{text}</span>;
  }
}

// GOOD: Composition over HOCs
function ProtectedPage({ children }: { children: React.ReactNode }) {
  return (
    <ErrorBoundary fallback={<ErrorPage />}>
      <AuthGuard>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </AuthGuard>
    </ErrorBoundary>
  );
}

// GOOD: Explicit props with TypeScript
interface ButtonProps {
  variant: 'primary' | 'secondary';
  size: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

function Button({ variant, size, disabled, onClick, children }: ButtonProps) {
  return (
    <button
      className={`btn btn-${variant} btn-${size}`}
      disabled={disabled}
      onClick={onClick}
    >
      {children}
    </button>
  );
}
```

**Guidelines:**
- **Never use `dangerouslySetInnerHTML` with user content**
- If HTML is required, sanitize with DOMPurify (whitelist approach)
- Never use `eval()` or `new Function()`
- Prefer composition over HOCs (max 1-2 HOCs if needed)
- Avoid spreading unknown props (`{...props}`)
- Make all component APIs explicit with TypeScript

---

### Rule 9: Type Safety — Strict TypeScript, No `any`

**Original Intent:** (C: Restrict pointer usage for safety)

**React Adaptation:**

```tsx
// BAD: Using any
function UserCard({ user }: { user: any }) {
  return <div>{user.name}</div>;  // No type safety
}

// BAD: Type assertions without validation
const user = response.data as User;  // Dangerous assumption

// BAD: Ignoring event types
function handleClick(e) {  // Implicit any
  console.log(e.target.value);
}

// GOOD: Strict component props
interface UserCardProps {
  user: User;
  onSelect?: (user: User) => void;
  className?: string;
}

function UserCard({ user, onSelect, className }: UserCardProps) {
  return (
    <div className={className} onClick={() => onSelect?.(user)}>
      {user.name}
    </div>
  );
}

// GOOD: Typed event handlers
function SearchInput({ onSearch }: { onSearch: (query: string) => void }) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onSearch(e.target.value);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    onSearch(formData.get('query') as string);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input name="query" onChange={handleChange} />
    </form>
  );
}

// GOOD: Generic hooks with type inference
function useLocalStorage<T>(key: string, initialValue: T) {
  const [value, setValue] = useState<T>(() => {
    const stored = localStorage.getItem(key);
    return stored ? (JSON.parse(stored) as T) : initialValue;
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue] as const;
}

// Usage: types are inferred
const [theme, setTheme] = useLocalStorage('theme', 'light');
//     ^? string

// GOOD: Discriminated unions for component variants
type ButtonProps =
  | { variant: 'link'; href: string; onClick?: never }
  | { variant: 'button'; onClick: () => void; href?: never };

function ActionButton(props: ButtonProps) {
  if (props.variant === 'link') {
    return <a href={props.href}>Click me</a>;
  }
  return <button onClick={props.onClick}>Click me</button>;
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
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true
  }
}
```

**Guidelines:**
- Enable all strict TypeScript options
- Never use `any` — use `unknown` and validate
- Type all event handlers explicitly
- Use generics for reusable hooks
- Use discriminated unions for variant props
- Infer types from Zod schemas for external data

---

### Rule 10: Static Analysis — ESLint React Hooks, Zero Warnings

**Original Intent:** Catch issues at development time; use every available tool.

**React Adaptation:**

```javascript
// eslint.config.js
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  {
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      // React Hooks — CRITICAL
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'error',

      // React Refresh (for HMR)
      'react-refresh/only-export-components': 'warn',

      // TypeScript strict
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-non-null-assertion': 'error',
      '@typescript-eslint/no-floating-promises': 'error',

      // Security
      'no-eval': 'error',
    },
  }
);
```

```tsx
// GOOD: React Strict Mode catches issues
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

```bash
# Required CI pipeline
tsc --noEmit                    # Type checking
eslint --max-warnings 0 src/    # Zero warnings policy
prettier --check src/           # Formatting
vitest run                      # Tests
```

**Error Boundary Setup:**
```tsx
// Use react-error-boundary for declarative error handling
import { ErrorBoundary } from 'react-error-boundary';

function App() {
  return (
    <ErrorBoundary
      fallback={<ErrorPage />}
      onError={(error, info) => {
        // Log to error tracking service
        console.error('Caught error:', error, info);
      }}
    >
      <MainContent />
    </ErrorBoundary>
  );
}
```

**Guidelines:**
- Enable `eslint-plugin-react-hooks` with both rules as errors
- Use React Strict Mode in development (catches unsafe lifecycles)
- Zero warnings policy — treat warnings as errors in CI
- Use Error Boundaries at route and feature boundaries
- Run `tsc --noEmit` in CI (catches type errors)
- No `// @ts-ignore` or `eslint-disable` without justification

---

## Summary: React Adaptation

| # | Original Rule | React Guideline |
|---|---------------|-----------------|
| 1 | No goto/recursion | No recursive components, guard clauses, flat trees |
| 2 | Fixed loop bounds | `.slice()` before map, virtualization for large lists |
| 3 | No dynamic allocation | Minimize useState, derive values, avoid useEffect for data |
| 4 | 60-line functions | 60-line components, extract custom hooks |
| 5 | 2+ assertions/function | TypeScript + Zod validation for external data |
| 6 | Minimize scope | Colocate state, composition over prop drilling |
| 7 | Check returns | Handle loading/error/empty states explicitly |
| 8 | Limit preprocessor | No dangerouslySetInnerHTML, composition over HOCs |
| 9 | Restrict pointers | Strict TypeScript, no `any`, typed event handlers |
| 10 | All warnings enabled | react-hooks/exhaustive-deps, StrictMode, zero warnings |

---

## References

- [Original Power of 10 Paper](https://spinroot.com/gerard/pdf/P10.pdf) — Gerard Holzmann
- [Rules of Hooks – React](https://react.dev/reference/rules/rules-of-hooks)
- [React 19 Release](https://react.dev/blog/2024/12/05/react-19)
- [eslint-plugin-react-hooks](https://www.npmjs.com/package/eslint-plugin-react-hooks)
- [react-error-boundary](https://github.com/bvaughn/react-error-boundary)
- [TanStack Virtual](https://tanstack.com/virtual/latest)
- [Zod Documentation](https://zod.dev/)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)
