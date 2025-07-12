# Next.js Technology Stack

## Core
Node.js: 20+ LTS
Package Manager: npm or pnpm
TypeScript: 5.0+ strict mode
Next.js: 14+ App Router only

## React
React: 19+
React DOM: 19+

## UI Components
shadcn/ui: Copy-paste components
Radix UI: Headless primitives
Tailwind CSS v4: Styling
clsx + tailwind-merge: Class utilities

## State Management
Zustand: Client state
TanStack Query v5: Server state

## Forms & Validation
react-hook-form: Form state management
zod: Schema validation
@hookform/resolvers: Zod integration

## Dynamic Tables
@tanstack/react-table: Table rendering
@tanstack/virtual: Virtual scrolling

## Schema Translation Tools (Under Evaluation)
### Option 1: Direct Translation
pydantic2zod: Direct Pydantic → Zod conversion

### Option 2: JSON Schema Bridge
json-schema-to-zod: JSON Schema → Zod code generation
zod-from-json-schema: Runtime JSON Schema → Zod
@rjsf/core: JSON Schema → Dynamic forms
@rjsf/validator-ajv8: JSON Schema validation

### Option 3: OpenAPI Based
openapi-typescript: OpenAPI → TypeScript types
openapi-fetch: Type-safe API client

Note: Best approach TBD based on project requirements

## Routing & Navigation
Next.js App Router: Built-in
next/navigation: Hooks
next/link: Client navigation

## Data Fetching
fetch: Native with Next.js cache
swr: Client-side caching (if needed)

## Testing
Jest: Test runner
React Testing Library: Component tests
@testing-library/user-event: User interactions
Playwright: E2E testing

## Build Tools
Turbo: Monorepo management
ESLint 9: Linting (flat config)
Prettier: Code formatting
lint-staged: Pre-commit formatting
husky: Git hooks

## Development Tools
@types/node: Node types
@types/react: React types
typescript-eslint: TS linting

## Styling
Tailwind CSS: Utility classes
PostCSS: CSS processing
@tailwindcss/typography: Prose styles

## Icons & Assets
lucide-react: Icon library
next/image: Image optimization
@vercel/og: OG image generation

## Analytics & Monitoring
@vercel/analytics: Web analytics
@vercel/speed-insights: Performance

## Utilities
date-fns: Date formatting
class-variance-authority: Component variants
cmdk: Command menu
sonner: Toast notifications

## Animation
framer-motion: Complex animations
@radix-ui/react-dialog: Animated modals

## Data Grid (Advanced)
ag-grid-react: Enterprise data grids (when needed)

## BANNED - NEVER USE
- Create React App
- Redux, MobX
- Material-UI, Ant Design, Chakra UI
- styled-components, emotion
- Formik
- Moment.js
- axios (use fetch or openapi-fetch)
- lodash (use native JS)
- Bootstrap, Foundation
- yup, joi (use zod)
- react-table v7 (use @tanstack/react-table)

## Schema Flow Pattern

### Backend → Frontend
1. Pydantic model → JSON Schema/OpenAPI
2. Choose translation strategy based on project
3. Generate types and validation schemas
4. Render dynamic forms and tables

### Frontend → Backend
1. Form data validated with zod
2. Type-safe API calls
3. Response handling with generated types

## Exceptions
Project-specific packages require explicit documentation in project README.