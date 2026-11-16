# Frontend

Next.js (App Router) + TypeScript + Tailwind. **Khởi tạo đầy đủ vào Ngày 19** bằng `create-next-app`,
tránh nửa cấu hình hỏng ở giai đoạn scaffolding.

## Ngày 19 sẽ làm
```bash
npx create-next-app@latest . --typescript --tailwind --app --eslint
```
- `src/lib/supabase.ts` — `signInAnonymously()` zero-friction onboarding.
- `src/lib/api_client.ts` — đính kèm Bearer JWT.
- `public/fallback_feed.json` — hiển thị tức thì khi backend cold start.

## Cấu trúc đích (theo master plan)
```
src/app/page.tsx            Tab 1: Short-Script Feed
src/app/sandbox/page.tsx    Tab 2: Creator Sandbox Studio
src/app/dashboard/page.tsx  Tab 3: Engineering Dashboard
src/components/             ScriptCard, ProductRecommendation, MetricCharts, Navbar
src/lib/                    supabase.ts, api_client.ts
```