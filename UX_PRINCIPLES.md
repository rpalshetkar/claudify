# UX Design Principles

## Core Principles
1. **Clarity** - Function over form, always
2. **Speed** - Performance is a feature
3. **Consistency** - Same patterns everywhere
4. **Feedback** - Every action has a response
5. **Accessibility** - WCAG 2.1 AA minimum

## Design Research Process
1. **Screenshot Analysis** - Search existing solutions first
2. **Pattern Matching** - Find 3-5 similar implementations
3. **Theme Detection** - Identify formal vs casual context
4. **Best Practices** - Extract what works

### Theme Auto-Detection
- **Formal/Enterprise**: Banking, healthcare, government
  - Conservative colors
  - Clear hierarchies
  - Explicit labeling
  - Dense information
  
- **Casual/Consumer**: Social, entertainment, lifestyle
  - Playful interactions
  - Conversational copy
  - Gesture-based
  - Minimal text

## Internationalization

### String Management
- All text in locale files
- Keys not English phrases: `button.submit` not `button.Submit`
- Pluralization support built-in
- RTL languages considered
- No hardcoded strings EVER

### Locale Structure
```
locales/
├── en-US.json
├── es-ES.json
├── ar-SA.json (RTL)
└── zh-CN.json
```

### String Keys Pattern
```json
{
  "nav.home": "Home",
  "form.email.label": "Email Address",
  "form.email.error.invalid": "Please enter a valid email",
  "action.save": "Save",
  "status.loading": "Loading..."
}
```

## Visual Hierarchy

### Typography
- System font stack (speed first)
- 3 sizes maximum (sm, base, lg)
- Line height 1.5+ for readability
- Contrast ratio 4.5:1 minimum

### Colors
- 5 colors max (primary, secondary, error, warning, success)
- Semantic naming (not red/green)
- Dark mode from day one
- Test with color blindness simulator

### Spacing
- 8px grid system
- Consistent padding: 8, 16, 24, 32, 48
- Whitespace is not wasted space
- Mobile margins: 16px, Desktop: 24px

## Interaction Patterns

### Forms
- Labels above inputs (not placeholders)
- Real-time validation (on blur)
- Clear error messages with fixes
- Show field requirements upfront
- Progress indicator for multi-step

### Loading States
- Skeleton screens > Spinners
- Show progress percentage when possible
- Stagger animations (not all at once)
- Optimistic updates when safe

### Error Handling
- What went wrong (specific)
- How to fix it (actionable)
- Preserve user data always
- Offer retry or alternative path

### Buttons & Actions
- Primary action per screen
- Destructive actions need confirmation
- Disabled state with tooltip why
- Loading state inside button
- Touch target 44x44px minimum

## Performance Metrics
- LCP < 2.5s (Largest Contentful Paint)
- FID < 100ms (First Input Delay)
- CLS < 0.1 (Cumulative Layout Shift)
- TTI < 3.8s (Time to Interactive)

## Mobile First

### Touch Interactions
- Thumb zone consideration
- Swipe gestures where natural
- Tap targets 44x44px
- No hover-only interactions

### Responsive Breakpoints
- Mobile: 320px - 768px
- Tablet: 768px - 1024px
- Desktop: 1024px+
- Content drives breakpoints

## Accessibility Checklist
- [ ] Keyboard navigation works
- [ ] Screen reader tested
- [ ] Focus indicators visible
- [ ] ARIA labels where needed
- [ ] Color not sole indicator
- [ ] Text scalable to 200%
- [ ] Motion respects prefers-reduced-motion

## Data Display

### Tables
- Sticky headers on scroll
- Sortable columns indicated
- Responsive (cards on mobile)
- Bulk actions at top
- Pagination or infinite scroll

### Empty States
- Why it's empty
- What user can do
- Illustration (optional)
- Single clear action

### Data Density
- Progressive disclosure
- Summary → Details pattern
- Scannable, not readable
- Key info above fold

## Navigation

### Information Architecture
- 3 clicks to any page max
- Breadcrumbs for deep navigation
- Search when >20 items
- Clear active state

### Menu Patterns
- Mobile: Bottom nav or hamburger
- Desktop: Top nav or sidebar
- Max 7 top-level items
- Group related items

## Feedback & Confirmation

### Toast Notifications
- Top right or bottom center
- Auto-dismiss after 4s
- Action to undo when applicable
- Stack multiple, don't replace

### Modals & Dialogs
- Darken background
- Close on escape/outside click
- Focus trap inside
- Smooth enter/exit animations

## Copy & Microcopy

### Tone Detection
1. Analyze domain/industry
2. Check competitor language
3. Match user expectations
4. Formal vs casual automatically

### Voice Guidelines
- **Formal**: Professional, precise, complete sentences
- **Casual**: Friendly, conversational, contractions OK
- Never mix tones in same app
- Let context drive tone

### Copy Rules
- All text from locale files
- No inline strings
- Variables in strings supported
- Length variations considered

## Dark Mode

### Implementation
- CSS variables for colors
- Respect system preference
- Manual toggle available
- Smooth transition

### Considerations
- Reduce pure white/black
- Adjust shadows
- Test all color combinations
- Icons may need variants

## Testing Methods
1. 5-second test (first impression)
2. Task completion rate
3. Time to complete task
4. Error frequency
5. Accessibility audit
6. Performance metrics

## Anti-Patterns to Avoid
- Hardcoded strings
- Mystery meat navigation
- Infinite scroll without option
- Auto-playing media
- Blocking interstitials
- Confirm shaming
- False urgency
- Carousel for important content
- Hamburger menu on desktop
- English-only thinking