# Integration snippets — Reliability page

## 1. Import CSS in Reliability.tsx

Add at the top of `Reliability.tsx`:

```tsx
import "./Reliability.css";
```

## 2. React Router route (App.tsx or router config)

```tsx
import Reliability from "./pages/Reliability";

// Inside your <Routes>:
<Route path="/reliability" element={<Reliability />} />
```

## 3. Nav link (in your "À propos" nav section)

```tsx
import { NavLink } from "react-router-dom";

// In your nav component:
<NavLink
  to="/reliability"
  className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}
>
  Fiabilité
</NavLink>
```

## 4. Serve the manifest + assets

Copy `build/credibility/v3/` into your `public/credibility/v3/` folder
so the browser can fetch `/credibility/v3/manifest.json` and the images.

```bash
cp -r build/credibility/v3/ public/credibility/v3/
```

Or configure your dev server to serve `build/credibility/v3/` at `/credibility/v3/`.
