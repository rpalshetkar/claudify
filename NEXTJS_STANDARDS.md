# Next.js Code Standards

## Component Template
```typescript
'use client' // or 'use server' - ALWAYS explicit

import { type FC, type ReactNode } from 'react'
import { cn } from '@/lib/utils'

interface ComponentProps {
  className?: string
  children: ReactNode
}

export const Component: FC<ComponentProps> = ({ 
  className, 
  children 
}) => {
  return (
    <div className={cn('base-classes', className)}>
      {children}
    </div>
  )
}
```

## Hook Template
```typescript
import { useState, useCallback } from 'react'

interface UseExampleReturn {
  value: string
  setValue: (value: string) => void
  reset: () => void
}

export function useExample(initial = ''): UseExampleReturn {
  const [value, setValue] = useState(initial)
  
  const reset = useCallback(() => {
    setValue(initial)
  }, [initial])
  
  return { value, setValue, reset }
}
```

## Error Boundary
```typescript
'use client'

import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback: ReactNode
}

export class ErrorBoundary extends Component<Props> {
  state = { hasError: false }
  
  static getDerivedStateFromError() {
    return { hasError: true }
  }
  
  render() {
    if (this.state.hasError) {
      return this.props.fallback
    }
    return this.props.children
  }
}
```

## Loading Pattern
```typescript
import { Suspense } from 'react'
import { Skeleton } from '@/components/ui/skeleton'

export default async function Page() {
  return (
    <Suspense fallback={<Skeleton className="h-64" />}>
      <AsyncComponent />
    </Suspense>
  )
}
```

## Data Fetching (Server Component)
```typescript
async function getData() {
  const res = await fetch('https://api.example.com/data', {
    next: { revalidate: 3600 } // Cache 1 hour
  })
  
  if (!res.ok) {
    throw new Error('Failed to fetch')
  }
  
  return res.json()
}

export default async function Page() {
  const data = await getData()
  return <div>{data.title}</div>
}
```

## Code Rules

✅ DO:
- 'use client'/'use server' always explicit
- TypeScript strict mode
- Functional components only
- Error boundaries for all routes
- Loading states with Suspense
- Server components by default
- cn() for conditional classes
- Proper TypeScript types

❌ DON'T:
- Class components (except ErrorBoundary)
- any type
- useEffect for data fetching
- CSS-in-JS
- Inline styles
- Default exports for components
- Nested ternaries in JSX

## File Structure
```
src/
├── app/              # Next.js app router
├── components/
│   ├── ui/          # shadcn/ui components
│   └── features/    # Feature components
├── hooks/           # Custom hooks
├── lib/             # Utilities
└── types/           # TypeScript types
```

## Naming Conventions
Files: kebab-case.tsx
Components: PascalCase
Hooks: useXxx
Utils: camelCase
Types: PascalCase
Constants: UPPER_SNAKE_CASE

## Import Order
```typescript
// 1. React/Next
import { useState } from 'react'
import { useRouter } from 'next/navigation'

// 2. Third-party
import { format } from 'date-fns'

// 3. Components
import { Button } from '@/components/ui/button'

// 4. Utils/Hooks
import { cn } from '@/lib/utils'
import { useAuth } from '@/hooks/use-auth'

// 5. Types
import type { User } from '@/types'

// 6. Styles (if any)
import styles from './component.module.css'
```

## Testing Pattern
```typescript
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

describe('Component', () => {
  it('should handle click', async () => {
    const user = userEvent.setup()
    render(<Component />)
    
    await user.click(screen.getByRole('button'))
    expect(screen.getByText('Clicked')).toBeInTheDocument()
  })
})
```

## Form Pattern
```typescript
'use client'

import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'

const schema = z.object({
  email: z.string().email(),
  password: z.string().min(8)
})

type FormData = z.infer<typeof schema>

export function LoginForm() {
  const form = useForm<FormData>({
    resolver: zodResolver(schema)
  })
  
  const onSubmit = async (data: FormData) => {
    // Handle submission
  }
  
  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      {/* Form fields */}
    </form>
  )
}
```

## Config Files

### tsconfig.json
```json
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true
  }
}
```

### eslint.config.mjs
```javascript
import js from '@eslint/js'
import typescript from '@typescript-eslint/eslint-plugin'

export default [
  js.configs.recommended,
  {
    rules: {
      '@typescript-eslint/no-unused-vars': 'error',
      '@typescript-eslint/no-explicit-any': 'error'
    }
  }
]
```

### .prettierrc
```json
{
  "semi": false,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5"
}
```

## Verification Commands
```bash
npm run lint
npm run type-check
npm run test
npm run build
```