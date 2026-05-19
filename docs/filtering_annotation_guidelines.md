# Annotation Guidelines — AXTree candidate filtering

**Цель**: разметить кандидатов из `build_candidates.py` как **junk / not_junk / skip**, плюс пометить плохие страницы как **bad_page**. Эти labels пойдут в обучение классификатора, который будет применяться при inference на полном AXTree (вариант 2 — full-tree inference).

**Контекст для разметчика**: ты видишь screenshot страницы с красным прямоугольником вокруг кандидата + метаданные (role, text_len, num_descendants, link_density, text_subtree). Решение принимаешь по совокупности всех этих сигналов.

---

## Базовые определения

### junk
Кандидат, который **не несёт уникальной для страницы информации** и должен быть удалён из widget-tree representation. То есть это структурный или служебный мусор, который повторяется/служит навигации/делает технические утверждения.

**Типичные категории junk** (sub-labels, добавляются вторым кликом):
- `cookie` — cookie-banner, GDPR-консент, privacy popup
- `footer` — нижний копирайт, terms/privacy ссылки, лицензии, made-with
- `nav` — главное навигационное меню сайта (header-меню, breadcrumbs)
- `sidebar` — боковая колонка с навигацией, виджетами, related links (когда не несёт уникального контента)
- `social` — share-кнопки, follow-us, иконки соцсетей, embedded социальные виджеты
- `legal` — отдельные legal-блоки (privacy notice, ToS reminder, accessibility notice)
- `ad` — реклама, promo-блоки, sponsorship, newsletter signup
- `recurring` — блок, который повторяется на >20% страниц корпуса (если такое известно из hash-сигнала)
- `other` — junk не из списка выше

### not_junk
Кандидат, который **несёт уникальный для этой страницы контент**: основной текст, статья, продуктовая карточка, форма, контентная секция, hero-блок с осмысленной информацией, headline, body articles, comments (если они являются основной целью страницы), и т.п.

### skip
Не «непонятно что выбрать», а конкретные технические случаи:
- кандидат рендерится за пределами видимой области (offscreen, скрыт CSS-ом)
- кандидат пуст (text_subtree пустой и нет визуального содержимого)
- screenshot не загружается / повреждён
- bbox не соответствует видимому содержимому (артефакт рендеринга)

**Skip ≠ "не уверен"**. Если непонятно junk или not_junk — выбирай по правилам приоритетов ниже, не уходи в skip.

### bad_page
Вся страница не подлежит разметке. Используется когда:
- большинство кандидатов на странице не несут смысла (resolution selection полностью провален — нет осмысленных единиц)
- screenshot отсутствует или дисплей-broken
- страница на языке, который не получается интерпретировать (CJK без графической интуиции)
- страница вылетает за все предположения (parking page, 404 без основного контента, captcha)

**bad_page применяется ко всей странице сразу** — это сигнал «эта page бесполезна для всего pipeline'а», а не к отдельному кандидату.

---

## Правила приоритета (для пограничных случаев)

Если кандидат может быть и junk, и not_junk одновременно, применяй правила в порядке:

1. **Если кандидат — это очевидная UI-конструкция, повторяющаяся между страницами сайта** (хедер, футер, nav-меню, cookie-banner, share-кнопки) → **junk**, даже если внутри есть уникальный текст. Уникальность отдельных слов в шаблоне не делает шаблон уникальным.

2. **Если кандидат — это основной контент страницы** (article body, product description, форма, главный заголовок) → **not_junk**, даже если он маленький.

3. **Если кандидат — это контейнер, охватывающий И junk И not_junk** (coarse granularity, частая проблема генератора): смотри на доминирующее содержимое.
   - Если >70% площади / текста — junk → размечай как **junk** соответствующего типа
   - Если >70% площади / текста — content → размечай как **not_junk**
   - Если действительно ~50/50 → **skip** + рассмотри возможность пометить страницу как `bad_page`, если такая coarse-проблема системная

4. **Если внутри кандидата основной контент, но с примесью social-кнопок / share-bars** → обычно **not_junk** (главное доминирует, social — небольшая примесь). Sub-классификация уровня social-кнопки достанется ML.

5. **Если кандидат — пустой layout-wrapper** (text_subtree пустой, num_descendants=0, или есть детей, но они тоже пусты) → **junk** sub-label `other`. Это шум структуры.

6. **Если кандидат — это header-секция со логотипом и заголовком статьи**: header нав-меню → junk, hero-блок с заголовком → not_junk. Если они склеены в один кандидат (coarse) → правило 3.

---

## Конкретные sub-кейсы

### Hero / Banner блоки
- Hero с CTA-кнопками + tagline на marketing-странице → **not_junk** (это первичный контент маркетинга)
- Hero с share-кнопками и copyright рядом → правило 3

### Navigation / Breadcrumbs
- Главное меню сайта (top nav) → **junk** sub-label `nav`
- Breadcrumbs внизу-вверху → **junk** sub-label `nav`
- Sidebar nav в documentation-сайте, который содержит ToC → **not_junk** (это ToC текущей доки)
- Footer-nav с links (about/contact/jobs) → **junk** sub-label `nav`

### Comments / User content
- Comments под статьёй (если статья — главный контент) → **not_junk** (вторичный, но реальный user content; решение неоднозначное, но я бы оставил not_junk)
- Single "Comments (5)" заголовок без самих комментов → **junk** sub-label `other`

### Forms
- Контактная форма / login form / search → **not_junk** (это функциональный UI элемент)
- Newsletter signup form → **junk** sub-label `ad` (это lead generation, не контент)

### Search / Filter widgets
- Search input + button → **not_junk** (функциональный UI)
- Filter sidebar в e-commerce → **not_junk** (функциональный UI)

### Images / Media
- Hero image с подписью → **not_junk**
- Decorative banner image без подписи → **junk** sub-label `other`
- Thumbnail-карточка статьи в листинге → **not_junk** (это и есть контент списка)

### Cookie / Privacy banners
- Любой cookie-banner / GDPR-popup → **junk** sub-label `cookie`
- Newsletter "subscribe" inline в hero → **junk** sub-label `ad`

---

## Self-agreement protocol

После первых **200 размеченных кандидатов**:
1. Сделать паузу (минимум на ночь, оптимально сутки)
2. Открыть аннотатор заново и **повторно** разметить те же 200 кандидатов в новом проходе (записывая отдельно, не перезаписывая первый прогон)
3. Сравнить: agreement = доля совпадающих labels между двумя проходами
4. Целевой порог: **agreement ≥ 0.85**

Если ниже:
- Прочитать guidelines повторно, особенно правила приоритета
- Найти 5–10 расхождений и решить «правильное» значение для каждого, добавить в guidelines как edge case
- Размечать дальше с обновлёнными guidelines, **первые 200 переписать** на новых правилах
- Повторить self-agreement check после следующих 200

Цель — стабильный mental model, при котором ты на 100-м примере и на 1500-м решаешь одинаково.

---

## Что записывать в labels.jsonl

Текущий формат сохраняет: `page_id, node_id, label, timestamp, role, text_len`.

Расширенный формат (после задачи pre-flight #2+3):
```json
{
  "page_id": "1655890112189",
  "node_id": "8",
  "label": "junk",
  "sub_label": "nav",          // новое — для junk: cookie/footer/nav/sidebar/social/legal/ad/recurring/other
  "content_hash": "abc123...",  // новое — md5 от role + text_subtree, для recovery при перегенерации кандидатов
  "generator_version": "v1",    // новое — git commit hash или semver build_candidates.py
  "timestamp": "2026-05-09T10:23:45",
  "role": "navigation",
  "text_len": 603
}
```

`sub_label` пустой для not_junk и skip. `content_hash` всегда заполнен.
