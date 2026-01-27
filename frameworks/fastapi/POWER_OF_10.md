# The Power of 10 Rules for Safety-Critical FastAPI Code

## Background

The **Power of 10 Rules** were created in 2006 by **Gerard J. Holzmann** at NASA's Jet Propulsion Laboratory (JPL) Laboratory for Reliable Software. These rules were designed for writing safety-critical code in C that could be effectively analyzed by static analysis tools.

The rules were incorporated into JPL's institutional coding standard and used for major missions including the **Mars Science Laboratory** (Curiosity Rover, 2012).

> *"If these rules seem draconian at first, bear in mind that they are meant to make it possible to check safety-critical code where human lives can very literally depend on its correctness."* — Gerard Holzmann

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

## The Power of 10 Rules — FastAPI Edition

### Rule 1: Simple Control Flow — Async-Aware, Guard Clauses

**Original Intent:** Eliminate complex control flow that impedes static analysis.

**FastAPI Adaptation:**

```python
# BAD: Deeply nested handler
@app.post("/orders")
async def create_order(order: OrderInput, db: Session = Depends(get_db)):
    if order.items:
        if order.customer_id:
            customer = db.query(Customer).get(order.customer_id)
            if customer:
                if customer.is_active:
                    # Finally do something...
                    pass

# BAD: Blocking call in async route
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = db.query(User).get(user_id)  # Blocks event loop!
    return user

# GOOD: Guard clauses with early returns
@app.post("/orders", response_model=OrderResponse)
async def create_order(
    order: OrderInput,
    db: AsyncSession = Depends(get_async_db),
) -> OrderResponse:
    if not order.items:
        raise HTTPException(400, "Order must have items")

    customer = await db.get(Customer, order.customer_id)
    if not customer:
        raise HTTPException(404, "Customer not found")

    if not customer.is_active:
        raise HTTPException(400, "Customer account is inactive")

    created_order = await create_order_in_db(db, order)
    return OrderResponse.model_validate(created_order)

# GOOD: Use sync def for blocking operations
@app.get("/files/{file_id}")
def get_file(file_id: str):  # sync def - runs in threadpool
    with open(f"files/{file_id}", "rb") as f:
        return f.read()
```

**Guidelines:**
- Use guard clauses with `HTTPException` for early returns
- Maximum 3-4 levels of nesting
- Use `async def` only for truly async operations
- Use `def` (sync) for blocking I/O — FastAPI runs it in threadpool
- No recursion in route handlers

---

### Rule 2: Bounded Loops — Paginate and Limit

**Original Intent:** Ensure all loops terminate and can be analyzed statically.

**FastAPI Adaptation:**

```python
from fastapi import Query

# BAD: Unbounded query
@app.get("/users")
async def list_users(db: AsyncSession = Depends(get_async_db)):
    users = await db.execute(select(User))  # Could be millions
    return users.scalars().all()

# BAD: Unbounded request body
@app.post("/upload")
async def upload(file: UploadFile):
    content = await file.read()  # No size limit!

# GOOD: Paginated query with limits
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20

@app.get("/users", response_model=PaginatedResponse[UserResponse])
async def list_users(
    db: AsyncSession = Depends(get_async_db),
    page: int = Query(default=0, ge=0),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
) -> PaginatedResponse[UserResponse]:
    offset = page * page_size

    query = select(User).offset(offset).limit(page_size)
    result = await db.execute(query)
    users = result.scalars().all()

    return PaginatedResponse(
        data=[UserResponse.model_validate(u) for u in users],
        page=page,
        page_size=page_size,
    )

# GOOD: Limited file upload
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

@app.post("/upload")
async def upload(file: UploadFile):
    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Max size: {MAX_FILE_SIZE}")

    # Process content...

# GOOD: Bounded iteration
MAX_BATCH_SIZE = 1000

async def process_items(items: list[Item]) -> None:
    assert len(items) <= MAX_BATCH_SIZE, f"Batch too large: {len(items)}"

    for item in items[:MAX_BATCH_SIZE]:
        await process_item(item)
```

**Guidelines:**
- Always paginate database queries
- Use `Query(ge=, le=)` to validate pagination params
- Limit file upload sizes explicitly
- Use `itertools.islice` for unbounded iterators
- Assert collection sizes at boundaries

---

### Rule 3: Controlled Memory — Async DB, Stream Responses

**Original Intent:** Prevent unbounded memory growth.

**FastAPI Adaptation:**

```python
# BAD: Loading entire file into memory
@app.get("/download/{file_id}")
async def download(file_id: str):
    with open(f"files/{file_id}", "rb") as f:
        return Response(content=f.read())  # Entire file in memory

# BAD: Loading all records into memory
@app.get("/export")
async def export_users(db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(User))
    users = result.scalars().all()  # All users in memory
    return users

# GOOD: Stream large files
from fastapi.responses import StreamingResponse

@app.get("/download/{file_id}")
async def download(file_id: str):
    async def stream_file():
        async with aiofiles.open(f"files/{file_id}", "rb") as f:
            while chunk := await f.read(8192):
                yield chunk

    return StreamingResponse(stream_file(), media_type="application/octet-stream")

# GOOD: Stream database results
@app.get("/export")
async def export_users(db: AsyncSession = Depends(get_async_db)):
    async def stream_users():
        yield "["
        first = True
        async for user in await db.stream_scalars(select(User)):
            if not first:
                yield ","
            yield UserResponse.model_validate(user).model_dump_json()
            first = False
        yield "]"

    return StreamingResponse(stream_users(), media_type="application/json")

# GOOD: Use connection pooling
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)
async_session = async_sessionmaker(engine, expire_on_commit=False)
```

**Guidelines:**
- Use `StreamingResponse` for large files
- Use async database drivers (asyncpg, aiosqlite)
- Configure connection pool limits
- Use generators for large data exports
- Avoid storing request data in global state

---

### Rule 4: Short Handlers — Extract to Services

**Original Intent:** Ensure functions are small enough to understand and verify.

**FastAPI Adaptation:**

```python
# BAD: Monolithic handler
@app.post("/orders")
async def create_order(
    order: OrderInput,
    db: AsyncSession = Depends(get_async_db),
    current_user: User = Depends(get_current_user),
):
    # 200 lines of validation, business logic, database calls...
    pass

# GOOD: Handler delegates to service layer
@app.post("/orders", response_model=OrderResponse)
async def create_order(
    order: OrderInput,
    order_service: OrderService = Depends(get_order_service),
    current_user: User = Depends(get_current_user),
) -> OrderResponse:
    created_order = await order_service.create_order(
        user_id=current_user.id,
        order_input=order,
    )
    return OrderResponse.model_validate(created_order)

# Service layer (services/order_service.py)
class OrderService:
    def __init__(self, db: AsyncSession, inventory: InventoryService):
        self._db = db
        self._inventory = inventory

    async def create_order(
        self,
        user_id: UUID,
        order_input: OrderInput,
    ) -> Order:
        await self._validate_order(order_input)
        await self._reserve_inventory(order_input)
        order = await self._save_order(user_id, order_input)
        await self._notify_order_created(order)
        return order

    async def _validate_order(self, order: OrderInput) -> None:
        """≤60 lines"""
        ...

    async def _reserve_inventory(self, order: OrderInput) -> None:
        """≤60 lines"""
        ...

# Dependency injection
def get_order_service(
    db: AsyncSession = Depends(get_async_db),
    inventory: InventoryService = Depends(get_inventory_service),
) -> OrderService:
    return OrderService(db, inventory)
```

**Guidelines:**
- Maximum 60 lines per handler
- Extract business logic to service classes
- Use dependency injection for services
- One router file per domain (users, orders, etc.)
- Keep handlers focused on HTTP concerns

---

### Rule 5: Validation — Pydantic Models, Assert Invariants

**Original Intent:** Defensive programming catches bugs early.

**FastAPI Adaptation:**

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated

# GOOD: Strict Pydantic models with validators
class CreateUserInput(BaseModel):
    model_config = {"strict": True}

    email: Annotated[str, Field(min_length=5, max_length=255)]
    password: Annotated[str, Field(min_length=8, max_length=100)]
    name: Annotated[str, Field(min_length=1, max_length=50)]
    age: Annotated[int, Field(ge=0, le=150)] | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()

    @model_validator(mode="after")
    def validate_model(self) -> "CreateUserInput":
        # Cross-field validation
        return self

# GOOD: Validate path parameters with Annotated
from uuid import UUID

@app.get("/users/{user_id}")
async def get_user(
    user_id: Annotated[UUID, Path(description="User's unique identifier")],
    db: AsyncSession = Depends(get_async_db),
) -> UserResponse:
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return UserResponse.model_validate(user)

# GOOD: Assert invariants in services
class AccountService:
    async def transfer_funds(
        self,
        from_account: UUID,
        to_account: UUID,
        amount: Decimal,
    ) -> None:
        # Preconditions
        assert amount > 0, "Amount must be positive"
        assert from_account != to_account, "Cannot transfer to same account"

        from_balance = await self._get_balance(from_account)
        if from_balance < amount:
            raise InsufficientFundsError()

        await self._execute_transfer(from_account, to_account, amount)

        # Postcondition
        new_balance = await self._get_balance(from_account)
        assert new_balance >= 0, "Balance went negative"
```

**Guidelines:**
- Use Pydantic models with `strict=True`
- Add `Field()` constraints (min_length, ge, le, etc.)
- Use `@field_validator` for complex validation
- Use `Annotated` with `Path()`, `Query()`, `Body()`
- Assert preconditions and postconditions in services

---

### Rule 6: Minimal Scope — Dependency Injection, No Globals

**Original Intent:** Reduce state complexity and potential for misuse.

**FastAPI Adaptation:**

```python
# BAD: Global mutable state
db_connection = None  # Global!
cache = {}  # Global mutable dict!

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id in cache:
        return cache[user_id]
    user = db_connection.query(...)  # Using global

# BAD: Overly broad dependencies
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    app_state: AppState = Depends(get_app_state),  # Everything!
):
    # Handler has access to everything in app_state

# GOOD: Scoped dependencies
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session

# GOOD: Narrow dependencies - only what's needed
@app.get("/users/{user_id}")
async def get_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_async_db),  # Only DB
) -> UserResponse:
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(404)
    return UserResponse.model_validate(user)

# GOOD: Request-scoped state via dependency
class RequestContext:
    def __init__(self, request_id: str, user: User | None):
        self.request_id = request_id
        self.user = user

async def get_request_context(
    request: Request,
    user: User | None = Depends(get_current_user_optional),
) -> RequestContext:
    return RequestContext(
        request_id=request.headers.get("X-Request-ID", str(uuid4())),
        user=user,
    )
```

**Guidelines:**
- No module-level mutable state
- Use `Depends()` for all shared resources
- Keep dependencies narrow — only inject what's needed
- Use `@lru_cache` for singleton config
- Use request-scoped dependencies for request state

---

### Rule 7: Check Returns — Handle All Errors

**Original Intent:** Never ignore errors; verify at trust boundaries.

**FastAPI Adaptation:**

```python
# BAD: Unchecked database result
@app.get("/users/{user_id}")
async def get_user(user_id: UUID, db: AsyncSession = Depends(get_async_db)):
    user = await db.get(User, user_id)
    return user  # Could be None!

# BAD: Bare except
@app.post("/users")
async def create_user(user: UserInput):
    try:
        return await user_service.create(user)
    except:  # Catches everything, including KeyboardInterrupt!
        return {"error": "Something went wrong"}

# GOOD: Explicit error handling with custom exceptions
class AppException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

class NotFoundError(AppException):
    def __init__(self, resource: str):
        super().__init__(404, f"{resource} not found")

class ValidationError(AppException):
    def __init__(self, detail: str):
        super().__init__(400, detail)

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

# GOOD: Check all returns explicitly
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_async_db),
) -> UserResponse:
    user = await db.get(User, user_id)
    if user is None:
        raise NotFoundError("User")

    return UserResponse.model_validate(user)

# GOOD: Specific exception handling
@app.post("/users", response_model=UserResponse)
async def create_user(
    user_input: UserInput,
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    try:
        user = await user_service.create(user_input)
    except IntegrityError:
        raise ValidationError("Email already exists")
    except ConnectionError:
        logger.exception("Database connection failed")
        raise HTTPException(503, "Service temporarily unavailable")

    return UserResponse.model_validate(user)
```

**Guidelines:**
- Always check for `None` returns
- Define custom exception classes
- Use `@app.exception_handler` for consistent error responses
- Never use bare `except:`
- Log internal errors, return safe messages to clients

---

### Rule 8: Limit Metaprogramming — No eval, Simple Decorators

**Original Intent:** Avoid constructs that create unanalyzable code.

**FastAPI Adaptation:**

```python
# BAD: Dynamic code execution
@app.post("/query")
async def execute_query(query: str):
    result = eval(query)  # Never!
    return result

# BAD: Excessive decorator stacking
@app.get("/data")
@cache
@rate_limit
@log_request
@validate_token
@check_permissions
async def get_data():
    ...

# BAD: Dynamic route generation with exec
for entity in ["user", "product", "order"]:
    exec(f"""
@app.get("/{entity}s")
async def list_{entity}s():
    return await db.query({entity.title()}).all()
    """)

# GOOD: Simple middleware
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        logger.info(f"{request.method} {request.url.path} - {duration:.3f}s")
        return response

app.add_middleware(RequestLoggingMiddleware)

# GOOD: Explicit route factories
def create_crud_routes(
    router: APIRouter,
    model: type,
    response_model: type,
    service_dep: Callable,
) -> None:
    @router.get("/", response_model=list[response_model])
    async def list_items(service = Depends(service_dep)):
        return await service.list_all()

    @router.get("/{item_id}", response_model=response_model)
    async def get_item(item_id: UUID, service = Depends(service_dep)):
        return await service.get(item_id)

# Usage is explicit and traceable
user_router = APIRouter(prefix="/users", tags=["users"])
create_crud_routes(user_router, User, UserResponse, get_user_service)
```

**Guidelines:**
- Never use `eval()` or `exec()`
- Limit decorator stacking to 2-3
- Use explicit route definitions over dynamic generation
- Use middleware classes over complex decorator chains
- Keep metaprogramming transparent and traceable

---

### Rule 9: Type Safety — Strict Pydantic, Type Hints Everywhere

**Original Intent:** (C: Restrict pointer usage for safety)

**FastAPI Adaptation:**

```python
from typing import Annotated, NewType
from pydantic import BaseModel, ConfigDict

# GOOD: Strict Pydantic configuration
class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

# GOOD: NewType for domain types
UserId = NewType("UserId", UUID)
Email = NewType("Email", str)

# GOOD: Typed response models
class UserResponse(StrictBaseModel):
    id: UserId
    email: Email
    name: str
    created_at: datetime

class PaginatedResponse(StrictBaseModel, Generic[T]):
    data: list[T]
    page: int
    page_size: int
    total: int | None = None

# GOOD: Fully typed handlers
@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: Annotated[UUID, Path(description="User ID")],
    db: Annotated[AsyncSession, Depends(get_async_db)],
) -> UserResponse:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    return UserResponse.model_validate(user)

# GOOD: Typed dependencies
async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_async_db)],
) -> User:
    payload = decode_token(token)
    user = await db.get(User, payload.user_id)
    if user is None:
        raise HTTPException(401, "Invalid token")
    return user

CurrentUser = Annotated[User, Depends(get_current_user)]

@app.get("/me", response_model=UserResponse)
async def get_me(current_user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(current_user)
```

**pyproject.toml:**
```toml
[tool.mypy]
strict = true
plugins = ["pydantic.mypy"]

[tool.pydantic-mypy]
init_forbid_extra = true
init_typed = true
warn_required_dynamic_aliases = true
```

**Guidelines:**
- Use `strict=True` in Pydantic models
- Use `Annotated` for all dependencies
- Define `NewType` for domain identifiers
- Use `response_model` on all endpoints
- Enable mypy strict mode with Pydantic plugin

---

### Rule 10: Static Analysis — Ruff, mypy, Zero Warnings

**Original Intent:** Catch issues at development time.

**FastAPI Adaptation:**

```toml
# pyproject.toml
[tool.mypy]
strict = true
plugins = ["pydantic.mypy"]
disallow_untyped_defs = true
warn_return_any = true

[tool.ruff]
select = ["ALL"]
ignore = ["D203", "D213"]

[tool.ruff.per-file-ignores]
"tests/*" = ["S101"]  # Allow assert in tests

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

```bash
# Required CI pipeline
ruff check src/                  # Linting
ruff format --check src/         # Formatting
mypy src/                        # Type checking
pytest --cov=src tests/          # Tests with coverage
pip-audit                        # Dependency vulnerabilities
bandit -r src/                   # Security scanning
```

**Production Security:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI(
    # Disable docs in production
    docs_url=None if PRODUCTION else "/docs",
    redoc_url=None if PRODUCTION else "/redoc",
    openapi_url=None if PRODUCTION else "/openapi.json",
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

**Guidelines:**
- Run mypy in strict mode with Pydantic plugin
- Use ruff for linting and formatting
- Zero warnings policy
- Disable API docs in production
- Configure CORS and TrustedHost middleware
- Run `pip-audit` and `bandit` regularly

---

## Summary: FastAPI Adaptation

| # | Original Rule | FastAPI Guideline |
|---|---------------|-------------------|
| 1 | No goto/recursion | Guard clauses, async-aware (sync def for blocking) |
| 2 | Fixed loop bounds | Paginate queries, limit uploads |
| 3 | No dynamic allocation | StreamingResponse, connection pooling |
| 4 | 60-line functions | 60-line handlers, extract to services |
| 5 | 2+ assertions/function | Pydantic validation, assert invariants |
| 6 | Minimize scope | Depends() for all state, no globals |
| 7 | Check returns | Custom exceptions, explicit None handling |
| 8 | Limit preprocessor | No eval, simple middleware |
| 9 | Restrict pointers | Strict Pydantic, type hints everywhere |
| 10 | All warnings enabled | mypy strict, ruff, zero warnings |

---

## References

- [Original Power of 10 Paper](https://spinroot.com/gerard/pdf/P10.pdf) — Gerard Holzmann
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices)
- [How to Secure FastAPI](https://escape.tech/blog/how-to-secure-fastapi-api/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
